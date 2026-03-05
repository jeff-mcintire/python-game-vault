"""
agent.py — The core RPG agent.

Uses Claude's tool-use API to execute vault operations in a loop until
Claude decides it has finished the task.  Claude calls tools like
create_file / update_file / append_to_file; this module executes them
against the real filesystem, then feeds results back to Claude.
"""

import logging
from pathlib import Path
from typing import Optional

import anthropic

from vault import VaultManager
from embeddings import VaultIndex

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert RPG game master assistant and world-builder.
You manage an Obsidian markdown vault that stores everything about a tabletop RPG
campaign: characters, NPCs, locations, factions, shops, session notes, lore, and more.

You will be given:
1. A **Vault Structure** listing every .md file in the vault.
2. A set of **Relevant Files** (the most semantically similar to the user's request),
   shown with their full content.
3. The **User Request**.

Your job is to fulfill the request by calling the provided tools to create, update,
or append to vault files.  After all tool calls are complete, write a brief,
friendly summary of exactly what you did.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OBSIDIAN FORMATTING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Always include YAML frontmatter with at least: tags, type, and status.
• Use [[WikiLinks]] to cross-reference related notes.
• Use consistent #tags.
• Use heading levels (##, ###) to organise long files.

FILE PLACEMENT CONVENTIONS
  Characters/[Name].md
  NPCs/[Name].md
  Locations/[Region]/[Place Name].md
  Locations/[City]/Shops/[Shop Name].md
  Factions/[Name].md
  Sessions/Session-[###].md
  Lore/[Topic].md

IMPORTANT RULES
• When creating a new character or NPC, also check whether any related location,
  faction, or shop file should be updated with a reference.
• When something happens in a session, update affected character files
  (status, memories, relationships) in addition to the session note.
• Never delete files unless explicitly instructed — use update or append instead.
• Keep all content consistent with what is already in the vault.
"""


# ---------------------------------------------------------------------------
# Tool definitions (passed to Claude)
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "create_file",
        "description": (
            "Create a brand-new markdown file in the vault. "
            "Use for new characters, locations, NPCs, factions, shops, lore entries, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "relative_path": {
                    "type": "string",
                    "description": "Path relative to vault root, e.g. 'Characters/Mira Ashwood.md'",
                },
                "content": {
                    "type": "string",
                    "description": "Markdown body content (everything after the frontmatter).",
                },
                "frontmatter": {
                    "type": "object",
                    "description": "YAML frontmatter fields as a JSON object (tags, type, status, etc.).",
                },
            },
            "required": ["relative_path", "content"],
        },
    },
    {
        "name": "update_file",
        "description": (
            "Completely replace the content of an existing file. "
            "Use for major rewrites — e.g. updating a character sheet after significant events."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "relative_path": {
                    "type": "string",
                    "description": "Path relative to vault root.",
                },
                "content": {
                    "type": "string",
                    "description": "New full markdown body.",
                },
                "frontmatter": {
                    "type": "object",
                    "description": "Updated frontmatter dict.",
                },
            },
            "required": ["relative_path", "content"],
        },
    },
    {
        "name": "append_to_file",
        "description": (
            "Append new content to the end of an existing file without touching what is already there. "
            "Ideal for adding session memories, journal entries, or new inventory items."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "relative_path": {
                    "type": "string",
                    "description": "Path relative to vault root.",
                },
                "content": {
                    "type": "string",
                    "description": "Markdown content to append.",
                },
            },
            "required": ["relative_path", "content"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read the full content of a specific vault file. "
            "Use when you need details about a file that was NOT included in the context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "relative_path": {
                    "type": "string",
                    "description": "Path relative to vault root.",
                },
            },
            "required": ["relative_path"],
        },
    },
]


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class RPGAgent:
    def __init__(self, vault: VaultManager, index: VaultIndex, api_key: str):
        self.vault = vault
        self.index = index
        self.client = anthropic.Anthropic(api_key=api_key)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, prompt: str, top_k: int = 10) -> dict:
        """
        Execute an agent loop for the given prompt.

        Returns:
          {
            "response": str,
            "files_referenced": [str, ...],
            "files_modified": [str, ...],
            "operations_performed": [{"operation": ..., "path": ...}, ...]
          }
        """
        # 1. Semantic search → relevant files
        relevant_files = self.index.search(prompt, top_k=top_k)
        context = self._build_context(relevant_files)

        # 2. Initial message
        messages: list[dict] = [
            {
                "role": "user",
                "content": f"{context}\n\n---\n\n## User Request\n\n{prompt}",
            }
        ]

        files_referenced = [path for path, _ in relevant_files]
        files_modified: list[str] = []
        operations: list[dict] = []
        final_response = ""

        # 3. Agentic tool-use loop
        while True:
            response = self.client.messages.create(
                model="claude-opus-4-5",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            # Collect text + tool calls from this turn
            tool_calls = []
            for block in response.content:
                if block.type == "text":
                    final_response = block.text
                elif block.type == "tool_use":
                    tool_calls.append(block)

            # No tool calls or natural end → done
            if not tool_calls or response.stop_reason == "end_turn":
                break

            # 4. Execute each tool call
            tool_results = []
            for tc in tool_calls:
                result_text, meta = self._execute_tool(tc.name, tc.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": result_text,
                    }
                )
                operations.append(meta)
                op = meta.get("operation", "")
                path = meta.get("path")
                if op in ("create", "update", "append") and path:
                    if path not in files_modified:
                        files_modified.append(path)

            # 5. Feed results back for next turn
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        return {
            "response": final_response,
            "files_referenced": files_referenced,
            "files_modified": files_modified,
            "operations_performed": operations,
        }

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    def _build_context(self, relevant_files: list[tuple[str, float]]) -> str:
        parts = [
            "## Vault Structure\n```\n" + self.vault.get_structure() + "\n```",
            "\n## Relevant Files\n",
        ]
        for path, score in relevant_files:
            content = self.index.get_content(path)
            parts.append(
                f"### `{path}` *(relevance {score:.2f})*\n"
                f"```markdown\n{content}\n```\n"
            )
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_tool(self, tool_name: str, tool_input: dict) -> tuple[str, dict]:
        """Run a single tool call.  Returns (result_text, metadata_dict)."""
        try:
            if tool_name == "create_file":
                path = self.vault.write_file(
                    tool_input["relative_path"],
                    tool_input["content"],
                    tool_input.get("frontmatter"),
                )
                self._reindex_file(path)
                return f"✓ Created {path}", {"operation": "create", "path": path}

            elif tool_name == "update_file":
                path = self.vault.write_file(
                    tool_input["relative_path"],
                    tool_input["content"],
                    tool_input.get("frontmatter"),
                )
                self._reindex_file(path)
                return f"✓ Updated {path}", {"operation": "update", "path": path}

            elif tool_name == "append_to_file":
                path = self.vault.append_file(
                    tool_input["relative_path"],
                    tool_input["content"],
                )
                self._reindex_file(path)
                return f"✓ Appended to {path}", {"operation": "append", "path": path}

            elif tool_name == "read_file":
                rel = tool_input["relative_path"]
                content = self.vault.read_relative(rel)
                return content, {"operation": "read", "path": rel}

            else:
                return f"Unknown tool: {tool_name}", {
                    "operation": "error",
                    "tool": tool_name,
                }

        except Exception as e:
            logger.error(f"Tool '{tool_name}' failed: {e}", exc_info=True)
            return f"Error: {e}", {"operation": "error", "tool": tool_name}

    def _reindex_file(self, relative_path: str) -> None:
        """Update the embedding index after a write."""
        try:
            full_text = self.vault.read_relative(relative_path)
            self.index.update_file(relative_path, full_text)
        except Exception as e:
            logger.warning(f"Could not reindex {relative_path}: {e}")

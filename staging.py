"""
staging.py — In-memory staging area for proposed vault changes.

When the agent runs in dry-run mode, all write operations are captured
here instead of being committed to disk.  The staged changes are returned
to the user for review.  The user can then:

  • confirm  → all staged writes are committed to disk at once
  • modify   → agent reruns with the original prompt + user feedback
  • discard  → staging area is dropped, nothing touches the vault

Reads are served from the staging area first (so the agent can read a
file it just proposed to create/update), falling back to the real vault.
"""

import difflib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional

import frontmatter as fm_lib

from vault import VaultManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StagedChange:
    """A single proposed file operation, not yet committed."""

    operation: Literal["create", "update", "append", "delete"]
    relative_path: str
    proposed_content: str          # full final file text after operation
    original_content: Optional[str] = None  # what is on disk right now (None = new file)
    frontmatter_data: Optional[dict] = None

    @property
    def diff(self) -> str:
        """Unified diff of original → proposed content."""
        original_lines = (self.original_content or "").splitlines(keepends=True)
        proposed_lines = self.proposed_content.splitlines(keepends=True)
        delta = difflib.unified_diff(
            original_lines,
            proposed_lines,
            fromfile=f"a/{self.relative_path}",
            tofile=f"b/{self.relative_path}",
            lineterm="",
        )
        return "".join(delta) or "(no changes)"

    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "relative_path": self.relative_path,
            "proposed_content": self.proposed_content,
            "original_content": self.original_content,
            "diff": self.diff,
        }


@dataclass
class StagingArea:
    """
    Holds all proposed changes for a single agent run.

    Changes are keyed by relative_path.  If the agent touches the same
    file more than once (e.g. creates then appends), the later operation
    supersedes the earlier one in the staging area.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    original_prompt: str = ""
    agent_response: str = ""           # Claude's summary text
    files_referenced: list[str] = field(default_factory=list)

    # path → StagedChange
    _changes: dict[str, StagedChange] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Staging writes (called by agent instead of real vault methods)
    # ------------------------------------------------------------------

    def stage_write(
        self,
        vault: VaultManager,
        relative_path: str,
        content: str,
        frontmatter_data: Optional[dict],
    ) -> str:
        """Stage a create or update operation."""
        # Read current content if file already exists (for diff)
        original: Optional[str] = None
        try:
            original = vault.read_relative(relative_path)
        except FileNotFoundError:
            pass  # new file

        operation: Literal["create", "update"] = "update" if original is not None else "create"

        # Build the full proposed file text (with frontmatter if provided)
        proposed = _assemble(content, frontmatter_data)

        # If this path was already staged (e.g. created earlier this run),
        # keep the *original* from the first staged version for the diff.
        if relative_path in self._changes:
            original = self._changes[relative_path].original_content

        self._changes[relative_path] = StagedChange(
            operation=operation,
            relative_path=relative_path,
            proposed_content=proposed,
            original_content=original,
            frontmatter_data=frontmatter_data,
        )
        logger.info(f"[staging] staged {operation}: {relative_path}")
        return relative_path

    def stage_append(self, vault: VaultManager, relative_path: str, content: str) -> str:
        """Stage an append operation."""
        # Use already-staged content as the base if the file was already touched
        if relative_path in self._changes:
            current_base = self._changes[relative_path].proposed_content
            original = self._changes[relative_path].original_content
        else:
            try:
                current_base = vault.read_relative(relative_path)
                original = current_base
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Cannot append — file not found in vault: {relative_path}"
                )

        proposed = current_base.rstrip() + f"\n\n{content}"

        self._changes[relative_path] = StagedChange(
            operation="append",
            relative_path=relative_path,
            proposed_content=proposed,
            original_content=original,
        )
        logger.info(f"[staging] staged append: {relative_path}")
        return relative_path

    def stage_delete(self, vault: VaultManager, relative_path: str) -> str:
        """Stage a delete operation."""
        original: Optional[str] = None
        try:
            original = vault.read_relative(relative_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Cannot delete — file not found in vault: {relative_path}"
            )

        self._changes[relative_path] = StagedChange(
            operation="delete",
            relative_path=relative_path,
            proposed_content="",
            original_content=original,
        )
        logger.info(f"[staging] staged delete: {relative_path}")
        return relative_path

    # ------------------------------------------------------------------
    # Transparent reads
    # ------------------------------------------------------------------

    def read(self, vault: VaultManager, relative_path: str) -> str:
        """
        Return staged content if this file has been touched this run,
        otherwise fall through to the real vault.
        """
        if relative_path in self._changes:
            staged = self._changes[relative_path]
            if staged.operation == "delete":
                raise FileNotFoundError(
                    f"File staged for deletion: {relative_path}"
                )
            return staged.proposed_content
        return vault.read_relative(relative_path)

    # ------------------------------------------------------------------
    # Commit to disk
    # ------------------------------------------------------------------

    def commit(self, vault: VaultManager) -> list[str]:
        """
        Write all staged changes to the real vault.
        Returns list of paths that were actually modified.
        """
        committed: list[str] = []
        for path, change in self._changes.items():
            try:
                if change.operation in ("create", "update"):
                    vault.write_file(path, change.proposed_content)
                    committed.append(path)
                elif change.operation == "append":
                    # We stored the full proposed content, so overwrite is correct
                    vault.write_file(path, change.proposed_content)
                    committed.append(path)
                elif change.operation == "delete":
                    vault.delete_file(path)
                    committed.append(path)
            except Exception as e:
                logger.error(f"[staging] commit failed for {path}: {e}", exc_info=True)
        logger.info(f"[staging] committed {len(committed)} file(s)")
        return committed

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def changes(self) -> list[StagedChange]:
        return list(self._changes.values())

    @property
    def has_changes(self) -> bool:
        return bool(self._changes)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "original_prompt": self.original_prompt,
            "agent_response": self.agent_response,
            "files_referenced": self.files_referenced,
            "changes": [c.to_dict() for c in self.changes],
        }


# ---------------------------------------------------------------------------
# Session store  (in-memory; sessions expire after 30 minutes of inactivity)
# ---------------------------------------------------------------------------

class SessionStore:
    """Thread-safe dict of session_id → StagingArea, with TTL eviction."""

    SESSION_TTL_MINUTES = 30

    def __init__(self):
        self._sessions: dict[str, StagingArea] = {}

    def put(self, staging: StagingArea) -> None:
        self._sessions[staging.session_id] = staging
        self._evict()

    def get(self, session_id: str) -> Optional[StagingArea]:
        return self._sessions.get(session_id)

    def discard(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def _evict(self) -> None:
        """Remove sessions older than SESSION_TTL_MINUTES."""
        now = datetime.now(tz=timezone.utc)
        to_remove = []
        for sid, s in self._sessions.items():
            try:
                created = datetime.fromisoformat(s.created_at)
                age_min = (now - created).total_seconds() / 60
                if age_min > self.SESSION_TTL_MINUTES:
                    to_remove.append(sid)
            except Exception:
                pass
        for sid in to_remove:
            del self._sessions[sid]
            logger.info(f"[staging] evicted expired session {sid}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assemble(content: str, frontmatter_data: Optional[dict]) -> str:
    """Combine body + optional frontmatter into a full file text."""
    if not frontmatter_data:
        return content
    post = fm_lib.Post(content, **frontmatter_data)
    return fm_lib.dumps(post)

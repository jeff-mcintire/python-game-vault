# RPG Vault Agent — Backend

A Python/FastAPI backend that gives you Claude-powered natural language
control over your Obsidian RPG campaign vault — the same workflow as
`claude` CLI, exposed as a REST API you can connect any frontend to.

---

## How it works

```
Your Prompt
    │
    ▼
Semantic Search (sentence-transformers)
    │  Finds the most relevant .md files for your prompt
    ▼
Claude (tool-use loop)
    │  Reads context, decides which files to create/update/append
    ▼
File Operations (on your vault)
    │  Writes changes directly to your NAS/local filesystem
    ▼
Response + diff summary back to you
```

The vault is also watched with a `PollingObserver` so any changes you
make directly in Obsidian are automatically picked up and re-indexed.

---

## Setup

### 1. Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (install with `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Your NAS vault mounted at a local path (SMB/NFS/etc.)

### 2. Install dependencies

```bash
uv sync
```

That's it — `uv` creates the virtualenv and installs all dependencies automatically.

> The first run downloads the `all-MiniLM-L6-v2` embedding model (~80 MB).
> Subsequent starts load from cache.

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in ANTHROPIC_API_KEY and VAULT_PATH
```

### 4. Start the server

```bash
uv run uvicorn main:app --reload --port 8000
```

On first start the server will build the embedding index for your vault
(a few seconds per 100 files).  Subsequent starts load from `.vault_index.pkl`.

---

## API Reference

All write operations follow a **stage → review → confirm** workflow.
Nothing touches the vault until you explicitly confirm.

---

### `POST /chat` — Stage changes

Send a natural-language prompt. The agent runs fully but **nothing is written to disk yet**.

```json
{ "prompt": "Create a new NPC blacksmith named Aldric at the Iron Forge", "top_k": 10 }
```

**Response** (`PendingReview`):
```json
{
  "session_id": "a1b2c3d4-...",
  "agent_response": "I'll create Aldric Ironhand and update the Iron Forge...",
  "files_referenced": ["Locations/Stormhaven/Shops/Iron Forge.md"],
  "changes": [
    {
      "operation": "create",
      "relative_path": "NPCs/Aldric Ironhand.md",
      "proposed_content": "---\ntags: [npc]\n---\n## Aldric Ironhand\n...",
      "original_content": null,
      "diff": "--- a/NPCs/Aldric Ironhand.md\n+++ b/NPCs/Aldric Ironhand.md\n..."
    },
    {
      "operation": "update",
      "relative_path": "Locations/Stormhaven/Shops/Iron Forge.md",
      "proposed_content": "...",
      "original_content": "...",
      "diff": "--- a/...\n+++ b/...\n@@ ... @@\n..."
    }
  ],
  "operations_performed": [...]
}
```

Use the `session_id` to confirm, modify, or discard.

---

### `POST /review/{session_id}/confirm` — Commit to disk

Writes all staged changes to the vault. Cleans up the session.

```json
{
  "session_id": "a1b2c3d4-...",
  "files_committed": ["NPCs/Aldric Ironhand.md", "Locations/Stormhaven/Shops/Iron Forge.md"],
  "message": "Successfully committed 2 file(s) to vault."
}
```

---

### `POST /review/{session_id}/modify` — Request changes

Discards the current staged result and re-runs the agent with your original
prompt **plus** your feedback. Returns a fresh `PendingReview` with a new `session_id`.

```json
{ "feedback": "Give Aldric a backstory as a disgraced knight, not a lifelong smith" }
```

You can modify as many times as needed before confirming or discarding.

---

### `DELETE /review/{session_id}` — Discard

Drops the session. Nothing is written to the vault.

---

### `GET /status`

Returns index health and file counts.

### `GET /files?query=dragon+temple`

Without `query`: lists all indexed files.
With `query`: returns the top 20 semantically relevant files.

### `POST /reindex`

Triggers a full index rebuild in the background.

---

## Example prompts

```
"Create a new character sheet for Lady Vayne Mordecai, a cunning noble
 who secretly leads the Shadow Syndicate faction"

"Update Thorin's character file — he lost his left hand in last night's
 session fighting the Lich King's champion"

"Flesh out the Gilded Compass tavern in Saltmarsh — add the owner,
 3 regulars, a menu, and a rumour table"

"Log what happened in tonight's session: the party uncovered that
 Duke Aldran is the traitor, Mira levelled up to 7, and they found
 the Orb of Winters in the vault beneath the keep"

"Create a new faction: the Ember Court, a secret society of fire
 mages operating out of the volcanic Ashen Wastes"
```

---

## NAS / Network Share Notes

- Make sure the share is mounted **before** starting the server.
- The watcher uses `PollingObserver` (checks every 30 s) instead of
  inotify, which doesn't work over SMB/NFS.
- If your NAS is slow, increase `POLL_INTERVAL_SECONDS` in `watcher.py`.
- The embedding cache (`.vault_index.pkl`) is stored locally on the
  machine running this server, not on the NAS.

---

## Tuning

| Setting | Where | Default | Notes |
|---|---|---|---|
| `top_k` | per-request | 10 | Files pulled into Claude context |
| `EMBED_CHARS` | `embeddings.py` | 2000 | Chars embedded per file |
| `POLL_INTERVAL_SECONDS` | `watcher.py` | 30 | NAS polling frequency |
| Claude model | `agent.py` | `claude-opus-4-5` | Swap to sonnet for cheaper/faster |

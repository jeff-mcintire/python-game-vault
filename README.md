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

- Python 3.11+
- Your NAS vault mounted at a local path (SMB/NFS/etc.)

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> The first run downloads the `all-MiniLM-L6-v2` embedding model (~80 MB).
> Subsequent starts load from cache.

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in ANTHROPIC_API_KEY and VAULT_PATH
```

### 4. Start the server

```bash
uvicorn main:app --reload --port 8000
```

On first start the server will build the embedding index for your vault
(a few seconds per 100 files).  Subsequent starts load from `.vault_index.pkl`.

---

## API Reference

### `POST /chat`

Send a natural-language prompt to the agent.

```json
{
  "prompt": "Create a new NPC blacksmith named Aldric who works at the Iron Forge in Stormhaven",
  "top_k": 10
}
```

`top_k` controls how many semantically relevant files are pulled into
Claude's context window per request (default 10, max ~20 recommended).

**Response:**
```json
{
  "response": "Created Aldric Ironhand, blacksmith at the Iron Forge...",
  "files_referenced": ["Locations/Stormhaven/Shops/Iron Forge.md"],
  "files_modified": ["NPCs/Aldric Ironhand.md", "Locations/Stormhaven/Shops/Iron Forge.md"],
  "operations_performed": [
    {"operation": "create", "path": "NPCs/Aldric Ironhand.md"},
    {"operation": "update", "path": "Locations/Stormhaven/Shops/Iron Forge.md"}
  ]
}
```

### `GET /status`

Returns index health and file counts.

### `GET /files?query=dragon+temple`

Without `query`: lists all indexed files.
With `query`: returns the top 20 semantically relevant files for that query.

### `POST /reindex`

Triggers a full index rebuild in the background.
Use after bulk edits to the vault outside of this app.

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

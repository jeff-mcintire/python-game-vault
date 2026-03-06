# RPG Vault Agent — Backend

A Python/FastAPI backend that gives you natural language control over your
Obsidian RPG campaign vault. Supports multiple LLM providers (Claude and Grok)
for text operations, and Grok Aurora for AI image generation with vault-aware
prompt building.

---

## How it works

```
Your Prompt
    │
    ▼
Semantic Search (sentence-transformers)
    │  Finds the most relevant .md files for your prompt
    ▼
LLM Agent — Claude or Grok (tool-use loop)
    │  Reads context, decides which files to create/update/append
    ▼
Staging Layer
    │  All changes are staged for review — nothing touches disk yet
    ▼
Review → Confirm / Modify / Discard
    │  You approve before anything is written
    ▼
Vault (your NAS/local filesystem)
```

The vault is watched with a `PollingObserver` so any changes you make
directly in Obsidian are automatically picked up and re-indexed.

---

## Setup

### 1. Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Your NAS vault mounted at a local path (SMB/NFS/etc.)

### 2. Install dependencies

```bash
uv sync
```

`uv` creates the virtualenv and installs everything automatically.

> The first run downloads the `all-MiniLM-L6-v2` embedding model (~80 MB).
> Subsequent starts load from cache.

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set your keys and vault path:

```bash
# LLM Providers — set one or both
ANTHROPIC_API_KEY=sk-ant-...    # required for Claude
XAI_API_KEY=xai-...             # required for Grok chat + image generation

# Required
VAULT_PATH=/path/to/your/obsidian/vault
```

At least one LLM provider key is required to start. Both can be set
simultaneously — you choose which provider to use per request.

### 4. Start the server

```bash
uv run uvicorn main:app --reload --port 8000
```

On first start the server builds the embedding index for your vault
(a few seconds per 100 files). Subsequent starts load from `.vault_index.pkl`.

Interactive API docs are available at `http://localhost:8000/docs`.

---

## LLM Providers

The app supports two providers for all text/agent operations.
You select the provider per request — there is no global default beyond
what you specify in each call.

| Provider | Field value | Default model | Requires |
|---|---|---|---|
| Anthropic Claude | `"claude"` | `claude-opus-4-5` | `ANTHROPIC_API_KEY` |
| xAI Grok | `"grok"` | `grok-3` | `XAI_API_KEY` |

Image generation always uses Grok Aurora (`grok-imagine-image`) regardless
of the provider selected for chat operations, and requires `XAI_API_KEY`.

---

## API Reference

### Vault Agent Endpoints (Stage → Review → Confirm)

All write operations follow a **stage → review → confirm** workflow.
Nothing touches the vault until you explicitly confirm.

---

#### `POST /chat` — Stage changes

Send a natural-language prompt. The agent runs fully but **nothing is written
to disk yet**. Returns a `PendingReview` with a `session_id` you use to
confirm, modify, or discard.

**Request:**
```json
{
  "prompt": "Create a new NPC blacksmith named Aldric at the Iron Forge",
  "provider": "claude",
  "model": "claude-opus-4-5",
  "top_k": 10
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | required | Natural language instruction |
| `provider` | `"claude"` \| `"grok"` | `"claude"` | Which LLM to use |
| `model` | string | provider default | Override the provider's default model |
| `top_k` | integer | 10 | How many vault files to pull into context |

**Response** (`PendingReview`):
```json
{
  "session_id": "a1b2c3d4-...",
  "provider": "claude",
  "model": "claude-opus-4-5",
  "agent_response": "I'll create Aldric Ironhand and update the Iron Forge...",
  "files_referenced": ["Locations/Stormhaven/Shops/Iron Forge.md"],
  "changes": [
    {
      "operation": "create",
      "relative_path": "NPCs/Aldric Ironhand.md",
      "proposed_content": "---\ntags: [npc]\n---\n## Aldric Ironhand\n...",
      "original_content": null,
      "diff": "--- a/NPCs/Aldric Ironhand.md\n+++ b/NPCs/Aldric Ironhand.md\n..."
    }
  ]
}
```

---

#### `POST /review/{session_id}/confirm` — Commit to disk

Writes all staged changes to the vault. Cleans up the session.

```json
{
  "session_id": "a1b2c3d4-...",
  "files_committed": ["NPCs/Aldric Ironhand.md", "Locations/.../Iron Forge.md"],
  "message": "Successfully committed 2 file(s) to vault."
}
```

---

#### `POST /review/{session_id}/modify` — Request changes

Discards the current staged result and re-runs the agent with your original
prompt **plus** your feedback. Returns a fresh `PendingReview` with a new
`session_id`. You can optionally switch provider or model for the re-run.

```json
{
  "feedback": "Give Aldric a backstory as a disgraced knight, not a lifelong smith",
  "provider": "grok",
  "model": "grok-3-mini"
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `feedback` | string | required | What to change or improve |
| `provider` | `"claude"` \| `"grok"` | same as original | Switch provider for re-run |
| `model` | string | provider default | Switch model for re-run |

---

#### `DELETE /review/{session_id}` — Discard

Drops the session. Nothing is written to the vault.

---

### Utility Endpoints

#### `GET /status`

Returns index health, file count, and server uptime.

#### `GET /files?query=dragon+temple`

Without `query`: lists all indexed vault files.  
With `query`: returns the top 20 semantically relevant files for that query.

#### `POST /reindex`

Triggers a full index rebuild in the background. Useful after bulk vault changes.

#### `GET /providers`

Returns which providers are configured based on current environment variables,
along with their default and available models.

```json
{
  "providers": {
    "claude": {
      "configured": true,
      "default_model": "claude-opus-4-5"
    },
    "grok": {
      "configured": true,
      "default_model": "grok-3"
    }
  }
}
```

---

### Image Generation Endpoints

Both endpoints require `XAI_API_KEY` and use Grok Aurora (`grok-imagine-image`).
Each request returns **2 images by default** (configurable with `n`).

> Images are returned as temporary URLs hosted by xAI. Download them promptly
> if you want to keep them — they expire after a short window.
> Cost: **$0.07 per image** ($0.14 per default 2-image request).

#### Image Options

All image endpoints accept the same three optional display parameters:

| Field | Type | Default | Description |
|---|---|---|---|
| `aspect_ratio` | string | `"auto"` | Image proportions — see table below |
| `resolution` | `"1k"` \| `"2k"` | `"1k"` | Output resolution |
| `style` | string \| null | `null` | Art style preset — see table below |

**Supported aspect ratios:**

| Value | Best for |
|---|---|
| `auto` | Model picks the best ratio for the content |
| `1:1` | Social media, tokens, thumbnails |
| `16:9` | Widescreen scenes, location banners |
| `9:16` | Mobile, vertical story cards |
| `4:3` / `3:4` | Presentations, portrait cards |
| `3:2` / `2:3` | Photography style |
| `2:1` / `1:2` | Wide banners, headers |
| `19.5:9` / `9:19.5` | Modern smartphone displays |
| `20:9` / `9:20` | Ultra-wide scenes |

**Style presets** (appended to the prompt as text — not a native API parameter):

| Value | Effect |
|---|---|
| `photorealistic` | Ultra-photorealistic photography, sharp detail, natural lighting |
| `oil_painting` | Classical impressionism oil painting style |
| `watercolor` | Soft-edged watercolor with translucent washes |
| `pencil_sketch` | Detailed pencil sketch with cross-hatching and shading |
| `anime` | Cel-shaded anime illustration, vibrant colors |
| `dark_fantasy` | Dark fantasy digital art, dramatic lighting, highly detailed |
| `concept_art` | Professional concept art, matte painting, cinematic composition |
| `ink_wash` | East Asian brush art, minimal and expressive ink wash style |

---

#### `POST /images/generate` — Direct prompt

Generate images directly from your own prompt. The prompt is sent to Aurora
as-is (with the style suffix appended if `style` is set).

**Request:**
```json
{
  "prompt": "A hooded elven rogue on rain-slicked cobblestones at night, lantern glow",
  "n": 2,
  "aspect_ratio": "3:2",
  "resolution": "2k",
  "style": "dark_fantasy"
}
```

**Response:**
```json
{
  "images": [
    "https://images.x.ai/...",
    "https://images.x.ai/..."
  ],
  "prompt_used": "A hooded elven rogue on rain-slicked cobblestones at night, lantern glow — dark fantasy digital art, dramatic lighting, highly detailed",
  "aspect_ratio": "3:2",
  "resolution": "2k",
  "style": "dark_fantasy",
  "crafted_from_vault": false,
  "vault_files_used": []
}
```

---

#### `POST /images/from-vault` — Vault-aware generation

Describe what you want to visualize and optionally name specific vault files.
The app will:

1. Search the vault semantically for relevant content (characters, locations,
   factions, lore) related to your description.
2. Include any explicitly named `vault_references` files.
3. Ask Grok to craft a rich, lore-accurate image generation prompt from all
   that context — incorporating appearance details, setting, atmosphere, and
   the requested style.
4. Generate `n` images from the crafted prompt.

Returns the image URLs **and** the crafted prompt so you can see exactly what
was sent to Aurora.

**Request:**
```json
{
  "description": "Sable watching the docks at night from a rooftop",
  "vault_references": ["NPCs/Sable.md", "Locations/Dockward.md"],
  "top_k": 4,
  "n": 2,
  "aspect_ratio": "16:9",
  "resolution": "2k",
  "style": "dark_fantasy"
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `description` | string | required | What you want to visualize |
| `vault_references` | string[] | `[]` | Explicit vault file paths to include |
| `top_k` | integer | 6 | Number of semantic search results to also pull in |
| `n` | integer | 2 | Number of images to generate |
| `aspect_ratio` | string | `"auto"` | See image options above |
| `resolution` | string | `"1k"` | `"1k"` or `"2k"` |
| `style` | string \| null | `null` | Style preset — see image options above |

**Response:**
```json
{
  "images": [
    "https://images.x.ai/...",
    "https://images.x.ai/..."
  ],
  "prompt_used": "A lean half-elven woman with sharp amber eyes crouches on a rain-slicked rooftop above the Dockward's crowded wharves, her battered spyglass trained on a ship unloading crates under torchlight...",
  "aspect_ratio": "16:9",
  "resolution": "2k",
  "style": "dark_fantasy",
  "crafted_from_vault": true,
  "vault_files_used": [
    "NPCs/Sable.md",
    "Locations/Dockward.md",
    "Factions/Thieves Guild.md"
  ]
}
```

---

## Example Prompts

**Agent / vault writing:**
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

**Image generation:**
```
# Direct prompt — widescreen battle scene
POST /images/generate
{
  "prompt": "A desperate last stand at the gates of Valdris Keep, soldiers in silver armour fighting shadow creatures",
  "aspect_ratio": "16:9",
  "resolution": "2k",
  "style": "concept_art"
}

# Vault-aware — let the lore drive the image
POST /images/from-vault
{
  "description": "Aldric the Gray in his tower at midnight, surrounded by star charts",
  "top_k": 5,
  "aspect_ratio": "3:4",
  "style": "oil_painting"
}

# Character portrait
POST /images/from-vault
{
  "description": "Full portrait of House Valdris's heir",
  "vault_references": ["Characters/Valdris Heir.md", "Factions/House Valdris.md"],
  "aspect_ratio": "2:3",
  "resolution": "2k",
  "style": "photorealistic"
}
```

---

## NAS / Network Share Notes

- Mount the share **before** starting the server.
- The watcher uses `PollingObserver` (checks every 30 s) — inotify doesn't
  work over SMB/NFS.
- If your NAS is slow, increase `POLL_INTERVAL_SECONDS` in `watcher.py`.
- The embedding cache (`.vault_index.pkl`) is stored locally on the machine
  running this server, not on the NAS.

---

## Tuning

| Setting | Where | Default | Notes |
|---|---|---|---|
| `top_k` | per-request | 10 | Vault files pulled into agent context |
| `EMBED_CHARS` | `embeddings.py` | 2000 | Characters embedded per file |
| `POLL_INTERVAL_SECONDS` | `watcher.py` | 30 | NAS polling frequency (seconds) |
| Claude default model | `providers.py` | `claude-opus-4-5` | Change `CLAUDE_DEFAULT_MODEL` |
| Grok chat default model | `providers.py` | `grok-3` | Change `GROK_DEFAULT_MODEL` |
| Image model | `image_gen.py` | `grok-imagine-image` | Change `IMAGE_MODEL` |
| Image count | per-request | `2` | Set `n` in the request body |

---

## Project Structure

```
python-game-vault/
├── main.py           — FastAPI app, all endpoints
├── agent.py          — Provider-agnostic LLM agent loop
├── providers.py      — Claude + Grok provider implementations + factory
├── image_gen.py      — Grok Aurora image generation + vault prompt builder
├── models.py         — Pydantic request/response models
├── embeddings.py     — Sentence-transformer vault index
├── staging.py        — In-memory staging area + session store
├── vault.py          — Vault file read/write operations
├── watcher.py        — PollingObserver for auto-reindex on file change
├── pyproject.toml    — Dependencies (uv)
└── .env.example      — Environment variable template
```
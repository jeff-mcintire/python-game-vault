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
XAI_API_KEY=xai-...             # required for Grok chat + image/video generation

# fal.ai — required for enhancement tools (upscaling, etc.)
FAL_KEY=...                     # get from fal.ai/dashboard/keys

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
| xAI Grok | `"grok"` | `grok-4-1-fast-reasoning` | `XAI_API_KEY` |

**Available Grok chat models:**

| Model | Notes |
|---|---|
| `grok-4-1-fast-reasoning` | Default — fast with reasoning capability |
| `grok-4-1-fast-non-reasoning` | Fast, no reasoning overhead — best for simple creative tasks |
| `grok-3` | Previous flagship |
| `grok-3-mini` | Lightweight / low-cost option |

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

Both endpoints support two image generation backends selectable via the `provider` field.

| Provider | Model | Style | URL lifetime | Cost | Requires |
|---|---|---|---|---|---|
| `aurora` (default) | `grok-imagine-image` | Photorealistic, cinematic | ⚠️ Temporary | $0.07/image | `XAI_API_KEY` |
| `flux2pro` | `fal-ai/flux-2-pro` | Illustrated, painterly, stylised | ✅ Persistent | $0.03/MP | `FAL_KEY` |

**When to use each:**
- **Aurora** — photorealistic character portraits, dramatic cinematic scenes, realistic maps
- **FLUX 2 Pro** — illustrated RPG art, painterly environments, stylised concept art, anything where you want that "fantasy book cover" look rather than a photograph

> ⚠️ Aurora URLs are temporary — download promptly. FLUX 2 Pro URLs are persistent fal.media links.

Each request returns **2 images by default** (configurable with `n`).

#### Image Options

All image endpoints accept the same optional display parameters:

| Field | Type | Default | Notes |
|---|---|---|---|
| `provider` | `"aurora"` \| `"flux2pro"` | `"aurora"` | Image generation backend |
| `aspect_ratio` | string | `"auto"` | See table below |
| `resolution` | `"1k"` \| `"2k"` | `"1k"` | Aurora only — ignored for FLUX 2 Pro |
| `style` | string \| null | `null` | Art style preset — see table below |
| `seed` | integer \| null | `null` | FLUX 2 Pro only — for reproducibility |
| `safety_tolerance` | `"1"`–`"6"` | `"2"` | FLUX 2 Pro only — use `"5"` for dark fantasy |

**Supported aspect ratios:**

| Value | Best for | FLUX 2 Pro mapping |
|---|---|---|
| `auto` | Model picks the best ratio | `landscape_4_3` |
| `1:1` | Social media, tokens, thumbnails | `square_hd` |
| `16:9` | Widescreen scenes, location banners | `landscape_16_9` |
| `9:16` | Mobile, vertical story cards | `portrait_16_9` |
| `4:3` / `3:4` | Presentations, portrait cards | `landscape_4_3` / `portrait_4_3` |
| `3:2` / `2:3` | Photography style | `landscape_4_3` / `portrait_4_3` |
| `2:1` / `1:2` | Wide banners, headers | `landscape_16_9` / `portrait_16_9` |
| `19.5:9` / `9:19.5` | Modern smartphone displays | `landscape_16_9` / `portrait_16_9` |
| `20:9` / `9:20` | Ultra-wide scenes | `landscape_16_9` / `portrait_16_9` |

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

Generate images directly from your own prompt.

**Aurora example:**
```json
{
  "prompt": "A hooded elven rogue on rain-slicked cobblestones at night, lantern glow",
  "provider": "aurora",
  "n": 2,
  "aspect_ratio": "3:2",
  "resolution": "2k",
  "style": "dark_fantasy"
}
```

**FLUX 2 Pro example:**
```json
{
  "prompt": "A hooded elven rogue on rain-slicked cobblestones at night, lantern glow",
  "provider": "flux2pro",
  "n": 2,
  "aspect_ratio": "3:2",
  "style": "oil_painting",
  "safety_tolerance": "5"
}
```

**Response:**
```json
{
  "images": [
    "https://storage.googleapis.com/falserverless/...",
    "https://storage.googleapis.com/falserverless/..."
  ],
  "prompt_used": "A hooded elven rogue on rain-slicked cobblestones at night, lantern glow — rendered as an oil painting in the style of classical impressionism",
  "provider": "flux2pro",
  "aspect_ratio": "3:2",
  "resolution": "1k",
  "style": "oil_painting",
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
4. Generate `n` images from the crafted prompt via the chosen `provider`.

> Note: Vault prompt crafting always uses Grok chat (requires `XAI_API_KEY`)
> regardless of which image backend renders the final image.

**Request:**
```json
{
  "description": "Sable watching the docks at night from a rooftop",
  "vault_references": ["NPCs/Sable.md", "Locations/Dockward.md"],
  "top_k": 4,
  "n": 2,
  "provider": "flux2pro",
  "aspect_ratio": "16:9",
  "style": "dark_fantasy",
  "safety_tolerance": "5"
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `description` | string | required | What you want to visualize |
| `vault_references` | string[] | `[]` | Explicit vault file paths to include |
| `top_k` | integer | 6 | Number of semantic search results to also pull in |
| `n` | integer | 2 | Number of images to generate |
| `provider` | string | `"aurora"` | `"aurora"` or `"flux2pro"` |
| `aspect_ratio` | string | `"auto"` | See image options above |
| `resolution` | string | `"1k"` | Aurora only |
| `style` | string \| null | `null` | Style preset — see image options above |
| `seed` | integer \| null | `null` | FLUX 2 Pro only |
| `safety_tolerance` | string | `"2"` | FLUX 2 Pro only — `"5"` for dark fantasy |

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

#### `POST /images/edit` — Grok Aurora Image Editing

Edit existing images using **grok-imagine-image** with a natural-language prompt.
The model understands image content and applies targeted changes while preserving
structure. Useful for style transfers, character adjustments, compositing, and
iterative multi-turn refinements.

**Requires** `XAI_API_KEY`.  Cost: **$0.022 per output image** ($0.02 output + $0.002 input).
Output URLs are temporary — download promptly.

> Note: The OpenAI SDK's `images.edit()` does **not** work here — xAI requires
> `application/json`, not `multipart/form-data`. This endpoint calls the xAI REST
> API directly.

**Single-image edit — style transfer:**
```json
{
  "prompt": "Render this as a dark fantasy oil painting with dramatic torchlight",
  "image_urls": ["https://images.x.ai/.../portrait.jpg"],
  "n": 1,
  "aspect_ratio": "3:4",
  "style": "oil_painting"
}
```

**Multi-image composite:**
```json
{
  "prompt": "Add the character from the first image into the tavern scene of the second",
  "image_urls": [
    "https://images.x.ai/.../character.jpg",
    "https://images.x.ai/.../tavern.jpg"
  ]
}
```

**Multi-turn chain** — pass the previous output URL as `image_urls[0]` to refine iteratively:
```json
{
  "prompt": "Now add a hooded cloak; keep everything else the same",
  "image_urls": ["https://images.x.ai/.../previous_output.jpg"]
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | required | What to change — be explicit about what to keep too |
| `image_urls` | string[] | required | 1–10 source image URLs or base64 data URIs |
| `n` | integer | 1 | Number of output variations to generate |
| `aspect_ratio` | string | `"auto"` | Output aspect ratio — see image options above |
| `resolution` | string | `"1k"` | `"1k"` or `"2k"` |
| `style` | string \| null | `null` | Style preset appended to the prompt |

**Response:**
```json
{
  "images": ["https://images.x.ai/.../edited.jpg"],
  "prompt_used": "Render this as a dark fantasy oil painting with dramatic torchlight — rendered as an oil painting in the style of classical impressionism",
  "source_count": 1,
  "aspect_ratio": "3:4",
  "resolution": "1k",
  "style": "oil_painting"
}
```

**Tips for best results:**
- Lock what shouldn't change: `"Keep the face, pose, and background unchanged; change only the armour."`
- For style transfers, describe the target fully rather than just naming the style.
- For multi-turn chains, reference the previous change to maintain continuity.
- Use `n: 2` to generate side-by-side variations of the same edit.

---

### Video Generation Endpoints

All three video endpoints require `XAI_API_KEY` and use `grok-imagine-video`.
Video generation is **asynchronous** — the server polls xAI internally and
returns the final result when ready. Typical wait: **30–90 seconds** depending
on duration and resolution.

> Videos are returned as temporary xAI-hosted URLs. **Download promptly.**
> Pricing is per second of generated video at the selected resolution.

#### Video Options

| Field | Type | Default | Description |
|---|---|---|---|
| `duration` | integer | `8` | Length in seconds (1–15). Ignored in edit mode. |
| `aspect_ratio` | string | `"16:9"` | See table below |
| `resolution` | string | `"720p"` | `480p` \| `720p` \| `1080p` |
| `style` | string \| null | `null` | Free-text camera/style direction (see below) |

**Supported aspect ratios:**

| Value | Best for |
|---|---|
| `16:9` | Default — widescreen landscape scenes |
| `9:16` | Vertical / mobile-portrait clips |
| `4:3` | Classic cinematic / tavern interiors |
| `3:4` | Portrait character reveals |
| `1:1` | Social, square format |

**Style (video)** is free-text, not a preset list. Describe camera and style
direction naturally — it's woven into the prompt before generation:

```
"slow push-in on a foggy night"
"sweeping aerial drone shot over the city"
"handheld shaky-cam chase through narrow alleys"
"epic slow-motion with lens flare"
"dark fantasy matte painting aesthetic"
```

---

#### `POST /videos/generate` — Text-to-video or Image-to-video

Generate a video from a text prompt. Optionally supply `image_url` to animate
a still image (image-to-video mode).

**Request (text-to-video):**
```json
{
  "prompt": "Stormhaven's harbor at dawn, fog rolling off the water, fishing boats setting out",
  "duration": 10,
  "aspect_ratio": "16:9",
  "resolution": "1080p",
  "style": "slow push-in, dark fantasy"
}
```

**Request (image-to-video):**
```json
{
  "prompt": "Bring the scene to life — torches flickering, cloaks billowing in the wind",
  "image_url": "https://your-server.com/sable_portrait.png",
  "duration": 8,
  "aspect_ratio": "9:16",
  "resolution": "720p"
}
```

**Response:**
```json
{
  "video_url": "https://vidgen.x.ai/.../video.mp4",
  "prompt_used": "Stormhaven's harbor at dawn… — slow push-in, dark fantasy",
  "duration": 10,
  "aspect_ratio": "16:9",
  "resolution": "1080p",
  "style": "slow push-in, dark fantasy",
  "request_id": "vg_abc123...",
  "moderated": true,
  "crafted_from_vault": false,
  "vault_files_used": []
}
```

---

#### `POST /videos/edit` — Edit an existing video

Modify an existing video by describing the desired changes. The source video
must be publicly accessible and **≤ 8.7 seconds**. Output length matches
the input — `duration` is not configurable in edit mode.

**Request:**
```json
{
  "video_url": "https://vidgen.x.ai/.../previous_clip.mp4",
  "prompt": "Make the sky stormy and overcast, add rain"
}
```

Common edit prompts:
- `"Restyle as dark fantasy concept art"`
- `"Remove the background figures, keep only the main character"`
- `"Add falling snow and frost to every surface"`
- `"Make it night — replace the daylight with moonlight and torches"`

---

#### `POST /videos/from-vault` — Vault-aware video generation

The vault-aware equivalent of `/images/from-vault`, built for cinematic
motion clips. Describe what you want to see; the agent searches the vault
for relevant lore, then Grok crafts a detailed cinematic prompt before
generating the video.

**Request:**
```json
{
  "description": "Sable fleeing across the Dockward rooftops at night",
  "vault_references": ["NPCs/Sable.md", "Locations/Dockward.md"],
  "top_k": 5,
  "duration": 10,
  "aspect_ratio": "16:9",
  "resolution": "1080p",
  "style": "handheld chase cam, dark and rainy"
}
```

**Response:**
```json
{
  "video_url": "https://vidgen.x.ai/.../video.mp4",
  "prompt_used": "A lean half-elven woman with amber eyes vaults between rain-slicked rooftops above the Dockward wharves, lantern-light catching her worn leather cloak as city guards shout below — handheld chase cam, close follow, dark fantasy tone, rain and mist.",
  "duration": 10,
  "aspect_ratio": "16:9",
  "resolution": "1080p",
  "style": "handheld chase cam, dark and rainy",
  "request_id": "vg_xyz789...",
  "moderated": true,
  "crafted_from_vault": true,
  "vault_files_used": ["NPCs/Sable.md", "Locations/Dockward.md", "Factions/Thieves Guild.md"]
}
```

---

#### `GET /videos/status/{request_id}` — Check job status

Returns the current status of any video generation job without blocking.
Useful if you want to submit a job and poll separately rather than waiting
on the blocking response.

```
GET /videos/status/vg_abc123...
```

```json
{
  "request_id": "vg_abc123...",
  "status": "done",
  "video_url": "https://vidgen.x.ai/.../video.mp4",
  "duration": 10,
  "moderated": true
}
```

Status values: `pending` | `done` | `error`

The `request_id` is always returned in every video response, so you can
use this endpoint to re-check any job after the fact.

---

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

**Video generation:**
```
# Text-to-video — atmospheric scene
POST /videos/generate
{ "prompt": "A hooded figure crossing a fog-drenched stone bridge at midnight, torches casting orange halos",
  "duration": 10, "aspect_ratio": "16:9", "resolution": "1080p",
  "style": "slow orbital pan, dark fantasy" }

# Image-to-video — animate an existing still
POST /videos/generate
{ "prompt": "Bring the character to life — cloak stirring in wind, eyes scanning the crowd",
  "image_url": "https://your-host.com/sable_portrait.png",
  "duration": 6, "aspect_ratio": "9:16" }

# Video edit — add weather to a previously generated clip
POST /videos/edit
{ "video_url": "https://vidgen.x.ai/.../previous.mp4",
  "prompt": "Turn this into a storm — dark clouds, driving rain, lightning in the distance" }

# Vault-aware — let the lore write the prompt
POST /videos/from-vault
{ "description": "Aldric the Gray performing the ritual in his tower",
  "top_k": 5, "duration": 12, "aspect_ratio": "16:9", "resolution": "1080p",
  "style": "slow push-in, arcane glow, horror undertones" }
```

---

### Enhancement & Moderation Endpoints (fal.ai)

Enhancement and moderation tools use [fal.ai](https://fal.ai) — a serverless GPU inference
platform hosting 600+ open-source models.  All fal.ai tools share the same
`FAL_KEY` credential and the same underlying call pattern, so adding more
tools later requires minimal code.

> Output files are stored on **fal.media** — unlike xAI image/video URLs
> which expire, fal.media URLs are persistent.

---

#### `POST /enhance/upscale` — Clarity Upscaler

Upscale and sharpen an image using **fal-ai/clarity-upscaler**.

Unlike a simple pixel resize, Clarity runs a Stable Diffusion + ControlNet
pipeline over the enlarged image, intelligently reconstructing fine detail
rather than just blowing up existing pixels. The `prompt` parameter guides
what detail gets added — passing the original generation prompt gives the
best results.

**Typical workflow:** Generate an image with Grok → immediately pipe the
URL into this endpoint → get back a permanent, high-res version.

**Request:**
```json
{
  "image_url": "https://images.x.ai/.../grok_output.png",
  "upscale_factor": 4,
  "prompt": "A hooded elven rogue on rain-slicked cobblestones — dark fantasy digital art",
  "creativity": 0.25,
  "resemblance": 0.75,
  "enable_safety_checker": false
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `image_url` | string | required | Any public image URL (including temporary xAI URLs) or base64 data URI |
| `upscale_factor` | float | `2.0` | Scale multiplier (2 = 2× resolution, 4 = 4×) |
| `prompt` | string | `"masterpiece, best quality, highres"` | Steers what detail gets added — use the original generation prompt |
| `negative_prompt` | string | `"(worst quality...)"` | What to exclude from added detail |
| `creativity` | float 0–1 | `0.35` | How much new detail to invent (low = faithful, high = more invented texture) |
| `resemblance` | float 0–1 | `0.6` | ControlNet strength — how closely to track original structure |
| `guidance_scale` | float | `4.0` | CFG scale — prompt adherence |
| `num_inference_steps` | int | `18` | Quality/speed tradeoff (10–50) |
| `seed` | int \| null | `null` | Set for reproducibility |
| `enable_safety_checker` | bool | `true` | Set `false` for dark fantasy / mature content |

**Creativity / resemblance guide for RPG art:**

| Content type | `creativity` | `resemblance` |
|---|---|---|
| Character portrait | 0.2–0.3 | 0.7–0.8 |
| Environment / scene | 0.4–0.6 | 0.4–0.6 |
| Map / diagram | 0.1 | 0.9 |

**Response:**
```json
{
  "image_url": "https://v3b.fal.media/files/.../upscaled.png",
  "width": 4096,
  "height": 4096,
  "file_size": 18204819,
  "content_type": "image/png",
  "seed": 42,
  "source_url": "https://images.x.ai/.../grok_output.png",
  "upscale_factor": 4.0
}
```

---

#### `POST /images/check-nsfw` — NSFW Checker

Binary NSFW / SFW classification using **fal-ai/x-ailab/nsfw**.

Accepts up to **10 image URLs** per request and returns a per-image verdict
(`true` = NSFW, `false` = SFW). Results are returned in the same order as
the input URLs.

**Requires** `FAL_KEY`.

**Intended workflow — generate with relaxed safety, then gate:**

```
POST /images/generate  { "provider": "flux2pro", "safety_tolerance": "5", "enable_safety_checker": false, ... }
  → returns image URLs

POST /images/check-nsfw  { "image_urls": [ ... ] }
  → returns is_nsfw verdict per image

Store / surface only images where is_nsfw == false
```

**Request:**
```json
{
  "image_urls": [
    "https://storage.fal.media/.../image1.jpg",
    "https://storage.fal.media/.../image2.jpg"
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `image_urls` | string[] | 1–10 image URLs or base64 data URIs to classify |

**Response:**
```json
{
  "results": [
    { "image_url": "https://storage.fal.media/.../image1.jpg", "is_nsfw": false },
    { "image_url": "https://storage.fal.media/.../image2.jpg", "is_nsfw": true }
  ],
  "total_checked": 2,
  "nsfw_count": 1,
  "sfw_count": 1
}
```

---

#### FLUX 2 Pro Safety Controls

FLUX 2 Pro (`provider: "flux2pro"`) exposes two independent safety controls:

| Field | Type | Default | Effect |
|---|---|---|---|
| `safety_tolerance` | `"1"`–`"6"` | `"2"` | Graduated filter — `1` = strictest, `6` = most permissive |
| `enable_safety_checker` | bool | `true` | Master switch — set `false` to disable the filter gate entirely |

**Recommended approach for RPG / dark fantasy content:**

- Start with `safety_tolerance: "5"` and `enable_safety_checker: true` — this handles the vast majority of dark fantasy art while keeping a safety net in place.
- If you're still getting false positives, set `enable_safety_checker: false` and rely on `POST /images/check-nsfw` as a post-generation gate instead.
- Setting both `safety_tolerance: "6"` and `enable_safety_checker: false` removes all generation-side filtering — use `check-nsfw` downstream if content policies matter.

> These controls are FLUX 2 Pro only — Aurora (Grok) uses its own internal moderation and does not expose equivalent parameters.

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
| Grok chat default model | `providers.py` | `grok-4-1-fast-reasoning` | Change `GROK_DEFAULT_MODEL` |
| Image model | `image_gen.py` | `grok-imagine-image` | Change `IMAGE_MODEL` |
| Image count | per-request | `2` | Set `n` in the request body |
| Video model | `video_gen.py` | `grok-imagine-video` | Change `VIDEO_MODEL` |
| Video poll interval | `video_gen.py` | `5` s | Change `DEFAULT_POLL_INTERVAL` |
| Video timeout | `video_gen.py` | `300` s | Change `DEFAULT_TIMEOUT_SECONDS` |
| Clarity upscaler model | `fal_tools.py` | `fal-ai/clarity-upscaler` | Change `FAL_CLARITY_MODEL` |
| FLUX 2 Pro model | `fal_tools.py` | `fal-ai/flux-2-pro` | Change `FAL_FLUX2_PRO_MODEL` |
| NSFW checker model | `fal_tools.py` | `fal-ai/x-ailab/nsfw` | Change `FAL_NSFW_MODEL` |

---

## Project Structure

```
python-game-vault/
├── main.py           — FastAPI app, all endpoints
├── agent.py          — Provider-agnostic LLM agent loop
├── providers.py      — Claude + Grok provider implementations + factory
├── image_gen.py      — Grok Aurora image generation + vault prompt builder
├── video_gen.py      — Grok Aurora video generation, editing + vault prompt builder
├── fal_tools.py      — fal.ai tool integrations (upscaling, FLUX 2 Pro, NSFW checking)
├── models.py         — Pydantic request/response models
├── embeddings.py     — Sentence-transformer vault index
├── staging.py        — In-memory staging area + session store
├── vault.py          — Vault file read/write operations
├── watcher.py        — PollingObserver for auto-reindex on file change
├── pyproject.toml    — Dependencies (uv)
└── .env.example      — Environment variable template
```
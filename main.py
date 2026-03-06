"""
main.py — FastAPI application entry point.

Workflow:
  1.  POST /chat                     → PendingReview (staged, nothing written)
  2a. POST /review/{id}/confirm      → CommitResponse (writes to disk)
  2b. POST /review/{id}/modify       → PendingReview (re-runs with feedback)
  2c. DELETE /review/{id}            → DiscardResponse (drops session)

Provider selection:
  Every /chat request includes a "provider" field: "claude" (default) or "grok".
  An optional "model" field overrides the provider's default model.
  /review/{id}/modify also accepts provider/model to switch mid-review.

Required env vars:
  ANTHROPIC_API_KEY   — for Claude
  XAI_API_KEY         — for Grok
  VAULT_PATH          — path to Obsidian vault
"""

# Path bootstrap - must be the first executable code in this file.
# When uvicorn --reload spawns a subprocess on Windows it does NOT inherit
# the parent sys.path, so subdirectory packages (providers/) are invisible.
# Inserting the project root here fixes it for every import that follows.
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agent import RPGAgent
from embeddings import VaultIndex
from fal_tools import clarity_upscale
from image_gen import build_vault_prompt, generate_images
from video_gen import (
    build_vault_video_prompt,
    edit_video,
    generate_video,
    poll_video,
)
from models import (
    ChatRequest,
    CommitResponse,
    DiscardResponse,
    FileListResponse,
    FileSearchResult,
    ImageGenerateRequest,
    ImageGenerateResponse,
    IndexStatus,
    ModifyRequest,
    OperationRecord,
    PendingReview,
    StagedChangeResponse,
    VaultImageRequest,
    VideoEditRequest,
    VideoGenerateRequest,
    VideoGenerateResponse,
    VideoStatusResponse,
    VaultVideoRequest,
    ClarityUpscaleRequest,
    ClarityUpscaleResponse,
)
from providers import ProviderName, create_provider
from staging import SessionStore, StagingArea
from vault import VaultManager
from watcher import start_watcher

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

_vault: VaultManager | None = None
_index: VaultIndex | None = None
_agent: RPGAgent | None = None
_sessions: SessionStore = SessionStore()
_watcher = None
_status: dict = {
    "total_files": 0,
    "indexed_files": 0,
    "last_indexed": None,
    "watching": False,
}


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _vault, _index, _agent, _watcher

    vault_path = os.getenv("VAULT_PATH")
    cache_path = os.getenv("INDEX_CACHE_PATH", ".vault_index.pkl")

    if not vault_path:
        raise RuntimeError("VAULT_PATH environment variable is not set.")

    # Validate at least one provider key is present
    has_claude = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_grok = bool(os.getenv("XAI_API_KEY"))
    if not has_claude and not has_grok:
        raise RuntimeError(
            "No LLM provider configured. Set ANTHROPIC_API_KEY and/or XAI_API_KEY."
        )

    _vault = VaultManager(vault_path)
    _index = VaultIndex(cache_path=cache_path)
    _agent = RPGAgent(_vault, _index)   # agent is provider-agnostic

    if not _index.load_cache():
        logger.info("No cache found — building embedding index from scratch…")
        files = _vault.scan_files()
        file_data = [_vault.read_file(f) for f in files]
        _index.build_index(file_data)
    _status["last_indexed"] = _now_iso()
    _status["total_files"] = len(_vault.scan_files())
    _status["indexed_files"] = _index.file_count

    _watcher = start_watcher(_vault, _index)
    _status["watching"] = True

    available = []
    if has_claude:
        available.append("claude")
    if has_grok:
        available.append("grok")
    logger.info(f"RPG Vault Agent ready. Available providers: {', '.join(available)}")
    yield

    if _watcher:
        _watcher.stop()
        _watcher.join()
        logger.info("Watcher stopped.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RPG Vault Agent",
    description=(
        "Claude/Grok-powered backend for managing your Obsidian RPG campaign vault. "
        "Choose your LLM provider per request via the `provider` field."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_agent():
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised.")


def _require_session(session_id: str) -> StagingArea:
    staging = _sessions.get(session_id)
    if staging is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found or has expired (TTL: 30 min).",
        )
    return staging


def _staging_to_response(staging: StagingArea, ops: list[dict]) -> PendingReview:
    return PendingReview(
        session_id=staging.session_id,
        agent_response=staging.agent_response,
        provider=ProviderName(staging.provider_name),
        model=staging.model_used,
        files_referenced=staging.files_referenced,
        changes=[
            StagedChangeResponse(
                operation=c.operation,
                relative_path=c.relative_path,
                proposed_content=c.proposed_content,
                original_content=c.original_content,
                diff=c.diff,
            )
            for c in staging.changes
        ],
        operations_performed=[OperationRecord(**op) for op in ops],
    )


def _run_agent(
    prompt: str,
    provider_name: ProviderName,
    model: str | None,
    top_k: int,
    staging: StagingArea,
) -> dict:
    """Instantiate the requested provider and run the agent."""
    try:
        provider = create_provider(provider_name, model=model)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    staging.provider_name = provider_name.value
    staging.model_used = provider.model

    return _agent.run(prompt, provider=provider, top_k=top_k, staging=staging)


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------

@app.post(
    "/chat",
    response_model=PendingReview,
    summary="Run the agent and return proposed changes for review",
)
async def chat(request: ChatRequest):
    """
    Runs the agent against your prompt using the specified provider.
    **Nothing is written to disk.**

    - `provider`: `"claude"` (default) or `"grok"`
    - `model`: optional override (e.g. `"grok-3-mini"`, `"claude-sonnet-4-5"`)

    Returns a `PendingReview` with a `session_id` for confirm/modify/discard.
    """
    _require_agent()
    try:
        staging = StagingArea(original_prompt=request.prompt)
        result = _run_agent(
            prompt=request.prompt,
            provider_name=request.provider,
            model=request.model,
            top_k=request.top_k,
            staging=staging,
        )
        staging.agent_response = result["response"]
        staging.files_referenced = result["files_referenced"]
        _sessions.put(staging)
        logger.info(
            f"Staged {len(staging.changes)} change(s) via {staging.provider_name}/{staging.model_used}"
            f" — session {staging.session_id}"
        )
        return _staging_to_response(staging, result["operations_performed"])
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Agent error during staging")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# POST /review/{session_id}/confirm
# ---------------------------------------------------------------------------

@app.post(
    "/review/{session_id}/confirm",
    response_model=CommitResponse,
    summary="Commit all staged changes to the vault",
)
async def confirm(session_id: str):
    _require_agent()
    staging = _require_session(session_id)

    if not staging.has_changes:
        _sessions.discard(session_id)
        return CommitResponse(session_id=session_id, files_committed=[], message="No changes to commit.")

    try:
        committed = staging.commit(_vault)
        for rel_path in committed:
            try:
                full_text = _vault.read_relative(rel_path)
                _index.update_file(rel_path, full_text)
            except Exception:
                pass  # deleted files won't be readable — that's fine
        _sessions.discard(session_id)
        logger.info(f"Committed {len(committed)} file(s) from session {session_id}")
        return CommitResponse(
            session_id=session_id,
            files_committed=committed,
            message=f"Successfully committed {len(committed)} file(s) to vault.",
        )
    except Exception as e:
        logger.exception(f"Commit failed for session {session_id}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# POST /review/{session_id}/modify
# ---------------------------------------------------------------------------

@app.post(
    "/review/{session_id}/modify",
    response_model=PendingReview,
    summary="Re-run the agent with modification feedback",
)
async def modify(session_id: str, request: ModifyRequest):
    """
    Discards the current staged changes and re-runs with your original
    prompt plus your feedback.

    Optionally switch provider or model for the re-run:
    ```json
    { "feedback": "Make him a former soldier", "provider": "grok", "model": "grok-3-mini" }
    ```
    """
    _require_agent()
    staging = _require_session(session_id)

    combined_prompt = (
        f"{staging.original_prompt}\n\n---\n\n**Revision request:** {request.feedback}"
    )

    # Use the requested provider/model, or fall back to what the session used
    provider_name = request.provider or ProviderName(staging.provider_name)
    model = request.model  # None = use provider default

    try:
        _sessions.discard(session_id)
        new_staging = StagingArea(original_prompt=combined_prompt)
        result = _run_agent(
            prompt=combined_prompt,
            provider_name=provider_name,
            model=model,
            top_k=10,
            staging=new_staging,
        )
        new_staging.agent_response = result["response"]
        new_staging.files_referenced = result["files_referenced"]
        _sessions.put(new_staging)
        logger.info(
            f"Modify re-run via {new_staging.provider_name}/{new_staging.model_used}"
            f" → {len(new_staging.changes)} staged change(s) — session {new_staging.session_id}"
        )
        return _staging_to_response(new_staging, result["operations_performed"])
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Agent error during modify")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# DELETE /review/{session_id}
# ---------------------------------------------------------------------------

@app.delete(
    "/review/{session_id}",
    response_model=DiscardResponse,
    summary="Discard staged changes without writing anything",
)
async def discard(session_id: str):
    found = _sessions.discard(session_id)
    if not found:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    logger.info(f"Discarded session {session_id}")
    return DiscardResponse(
        session_id=session_id,
        message="Staged changes discarded. Nothing was written to the vault.",
    )


# ---------------------------------------------------------------------------
# GET /status
# ---------------------------------------------------------------------------

@app.get("/status", response_model=IndexStatus)
async def status():
    if _vault is None:
        raise HTTPException(status_code=503, detail="Not initialised.")
    _status["indexed_files"] = _index.file_count
    _status["total_files"] = len(_vault.scan_files())
    return IndexStatus(vault_path=str(_vault.vault_path), **_status)


# ---------------------------------------------------------------------------
# GET /providers
# ---------------------------------------------------------------------------

@app.get("/providers", summary="List configured LLM providers")
async def providers():
    """Returns which providers are available based on env vars set."""
    available = {}
    if os.getenv("ANTHROPIC_API_KEY"):
        available["claude"] = {
            "default_model": "claude-opus-4-5",
            "other_models": ["claude-sonnet-4-5", "claude-haiku-4-5-20251001"],
        }
    if os.getenv("XAI_API_KEY"):
        available["grok"] = {
            "default_model": "grok-3",
            "other_models": ["grok-3-mini", "grok-2"],
        }
    return {"providers": available}


# ---------------------------------------------------------------------------
# GET /files
# ---------------------------------------------------------------------------

@app.get("/files", response_model=FileListResponse)
async def list_files(query: str = ""):
    if _index is None:
        raise HTTPException(status_code=503, detail="Not initialised.")
    if query:
        results = _index.search(query, top_k=20)
        return FileListResponse(files=[FileSearchResult(path=p, score=s) for p, s in results])
    return FileListResponse(
        files=[FileSearchResult(path=p, score=1.0) for p in _index.file_paths]
    )


# ---------------------------------------------------------------------------
# POST /reindex
# ---------------------------------------------------------------------------

@app.post("/reindex")
async def reindex(background_tasks: BackgroundTasks):
    if _vault is None or _index is None:
        raise HTTPException(status_code=503, detail="Not initialised.")

    def _do_reindex():
        logger.info("Full reindex started…")
        files = _vault.scan_files()
        file_data = [_vault.read_file(f) for f in files]
        _index.build_index(file_data)
        _status["total_files"] = len(files)
        _status["indexed_files"] = _index.file_count
        _status["last_indexed"] = _now_iso()
        logger.info(f"Reindex complete: {_index.file_count} files indexed.")

    background_tasks.add_task(_do_reindex)
    return {"message": "Reindex started in background. Check /status for progress."}


# ---------------------------------------------------------------------------
# POST /images/generate  — direct prompt → 2 images
# ---------------------------------------------------------------------------

@app.post(
    "/images/generate",
    response_model=ImageGenerateResponse,
    summary="Generate images directly from a text prompt",
)
async def images_generate(request: ImageGenerateRequest):
    """
    Generate images directly from your own prompt using Grok Aurora.

    Returns `n` image URLs (default 2). Images are hosted temporarily
    on xAI's storage — download them if you want to keep them.

    **Requires** `XAI_API_KEY` to be set.

    Example:
    ```json
    {
      "prompt": "A hooded elven rogue standing on rain-slicked cobblestones, lantern light, dark fantasy oil painting",
      "n": 2
    }
    ```
    """
    if not os.getenv("XAI_API_KEY"):
        raise HTTPException(status_code=400, detail="XAI_API_KEY is not set.")
    try:
        urls = generate_images(
            prompt=request.prompt,
            n=request.n,
            aspect_ratio=request.aspect_ratio,
            resolution=request.resolution,
            style=request.style,
        )
        return ImageGenerateResponse(
            images=urls,
            prompt_used=request.prompt,
            aspect_ratio=request.aspect_ratio,
            resolution=request.resolution,
            style=request.style,
            crafted_from_vault=False,
        )
    except Exception as e:
        logger.exception("Image generation failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# POST /images/from-vault  — vault-aware prompt builder + image generation
# ---------------------------------------------------------------------------

@app.post(
    "/images/from-vault",
    response_model=ImageGenerateResponse,
    summary="Generate images using vault lore to craft the prompt",
)
async def images_from_vault(request: VaultImageRequest):
    """
    Describe what you want to visualize and optionally name specific vault
    files. The app will:

    1. Search the vault semantically for relevant content (characters,
       locations, factions, lore) related to your description.
    2. Include any explicitly named `vault_references` files.
    3. Ask Grok to craft a rich, lore-accurate image generation prompt
       from all that context.
    4. Generate `n` images (default 2) from the crafted prompt.

    Returns the image URLs **and** the crafted prompt so you can see
    exactly what was sent to the image model.

    **Requires** `XAI_API_KEY` to be set.

    Example — referencing a vault character by name:
    ```json
    {
      "description": "Sable the half-elven informant watching the docks at night",
      "vault_references": ["NPCs/Sable.md", "Locations/Dockward.md"],
      "top_k": 4
    }
    ```

    Example — purely semantic, no explicit references:
    ```json
    {
      "description": "The throne room of House Valdris after the coup",
      "top_k": 6
    }
    ```
    """
    if not os.getenv("XAI_API_KEY"):
        raise HTTPException(status_code=400, detail="XAI_API_KEY is not set.")
    if _index is None or _vault is None:
        raise HTTPException(status_code=503, detail="Vault not initialised.")

    try:
        # 1. Semantic search for relevant vault files
        search_results = _index.search(request.description, top_k=request.top_k)
        files_used = [path for path, _ in search_results]

        # 2. Add any explicitly requested vault files (deduplicated)
        for ref in request.vault_references:
            if ref not in files_used:
                files_used.append(ref)

        # 3. Build vault context string from all selected files
        context_parts = []
        for path in files_used:
            try:
                content = _vault.read_relative(path)
                context_parts.append(f"### {path}\n{content}")
            except FileNotFoundError:
                logger.warning(f"Vault reference not found: {path}")

        vault_context = "\n\n".join(context_parts) if context_parts else "(no vault files found)"

        # 4. Ask Grok to craft a rich image prompt from the context
        crafted_prompt = build_vault_prompt(
            description=request.description,
            vault_context=vault_context,
            style=request.style,
        )

        # 5. Generate images from the crafted prompt
        urls = generate_images(
            prompt=crafted_prompt,
            n=request.n,
            aspect_ratio=request.aspect_ratio,
            resolution=request.resolution,
            style=request.style,
        )

        logger.info(
            f"Vault image: {len(files_used)} file(s) used, "
            f"{len(urls)} image(s) generated"
        )

        return ImageGenerateResponse(
            images=urls,
            prompt_used=crafted_prompt,
            aspect_ratio=request.aspect_ratio,
            resolution=request.resolution,
            style=request.style,
            crafted_from_vault=True,
            vault_files_used=files_used,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Vault image generation failed")
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================================================
# Video generation endpoints
# ===========================================================================

@app.post(
    "/videos/generate",
    response_model=VideoGenerateResponse,
    summary="Generate a video from a prompt (text-to-video or image-to-video)",
    tags=["Video Generation"],
)
async def videos_generate(request: VideoGenerateRequest):
    """
    Generate a short video clip using Grok Aurora (grok-imagine-video).

    **Text-to-video** (default): provide a `prompt` and optional `duration`,
    `aspect_ratio`, and `resolution`.

    **Image-to-video**: also supply `image_url` — the model will animate the
    still image guided by your prompt.

    This endpoint blocks until the video is ready (polls xAI internally).
    Typical wait time is 30–90 seconds depending on duration and resolution.

    Videos are returned as temporary xAI-hosted URLs — **download promptly**.
    Pricing is per second of generated video.
    """
    try:
        final_prompt = request.prompt
        if request.style:
            final_prompt = f"{request.prompt.rstrip('.')} — {request.style}"

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: generate_video(
                prompt=final_prompt,
                duration=request.duration,
                aspect_ratio=request.aspect_ratio,
                resolution=request.resolution,
                image_url=request.image_url,
            ),
        )

        logger.info(
            f"Video generated: {result['duration']}s | request_id={result['request_id']}"
        )

        return VideoGenerateResponse(
            video_url=result["video_url"],
            prompt_used=final_prompt,
            duration=result["duration"],
            aspect_ratio=request.aspect_ratio,
            resolution=request.resolution,
            style=request.style,
            request_id=result["request_id"],
            moderated=result["moderated"],
            crafted_from_vault=False,
            vault_files_used=[],
        )

    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        logger.exception("Video generation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/videos/edit",
    response_model=VideoGenerateResponse,
    summary="Edit an existing video by describing desired changes",
    tags=["Video Generation"],
)
async def videos_edit(request: VideoEditRequest):
    """
    Edit an existing video using Grok Aurora.

    Supply the URL of a video you want to change (must be publicly accessible,
    max **8.7 seconds**) and a `prompt` describing the desired edits.

    Examples:
    - `"Make the sky stormy and dark"`
    - `"Remove the background figures"`
    - `"Restyle as dark fantasy concept art"`
    - `"Add falling snow and frost to the scene"`

    Output duration matches the input video's length (not user-configurable
    in edit mode).  This endpoint blocks until the result is ready.
    """
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: edit_video(
                video_url=request.video_url,
                prompt=request.prompt,
            ),
        )

        logger.info(
            f"Video edit done: {result['duration']}s | request_id={result['request_id']}"
        )

        return VideoGenerateResponse(
            video_url=result["video_url"],
            prompt_used=request.prompt,
            duration=result["duration"],
            aspect_ratio="(from source)",
            resolution="(from source)",
            style=None,
            request_id=result["request_id"],
            moderated=result["moderated"],
            crafted_from_vault=False,
            vault_files_used=[],
        )

    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        logger.exception("Video edit failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/videos/from-vault",
    response_model=VideoGenerateResponse,
    summary="Generate a vault-aware video using campaign lore",
    tags=["Video Generation"],
)
async def videos_from_vault(request: VaultVideoRequest):
    """
    Vault-aware video generation.

    Describe what you want to see, optionally name specific vault files, and
    the agent will:

    1. Search the vault semantically for relevant characters, locations, lore.
    2. Load any explicitly listed `vault_references`.
    3. Ask Grok to craft a cinematic, lore-accurate video prompt from the context.
    4. Generate the video from the crafted prompt.

    The response includes `prompt_used` and `vault_files_used` so you can see
    exactly how the lore was interpreted.

    **Optional `style` field**: free-text camera/style direction that is woven
    into the vault prompt (e.g. `"slow push-in on a foggy night"`,
    `"sweeping aerial drone shot"`, `"handheld shaky-cam chase"`).
    """
    if _index is None or _vault is None:
        raise HTTPException(status_code=503, detail="Vault not initialised.")

    try:
        # 1. Semantic search
        search_results = _index.search(request.description, top_k=request.top_k)
        files_used = [path for path, _ in search_results]

        # 2. Explicit vault references (deduplicated)
        for ref in request.vault_references:
            if ref not in files_used:
                files_used.append(ref)

        # 3. Build vault context
        context_parts = []
        for path in files_used:
            try:
                content = _vault.read_relative(path)
                context_parts.append(f"### {path}\n{content}")
            except FileNotFoundError:
                logger.warning(f"Vault reference not found: {path}")

        vault_context = "\n\n".join(context_parts) if context_parts else "(no vault files found)"

        # 4. Craft cinematic prompt via Grok
        crafted_prompt = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: build_vault_video_prompt(
                description=request.description,
                vault_context=vault_context,
                style=request.style,
            ),
        )

        # 5. Generate video
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: generate_video(
                prompt=crafted_prompt,
                duration=request.duration,
                aspect_ratio=request.aspect_ratio,
                resolution=request.resolution,
            ),
        )

        logger.info(
            f"Vault video: {len(files_used)} file(s), "
            f"{result['duration']}s | request_id={result['request_id']}"
        )

        return VideoGenerateResponse(
            video_url=result["video_url"],
            prompt_used=crafted_prompt,
            duration=result["duration"],
            aspect_ratio=request.aspect_ratio,
            resolution=request.resolution,
            style=request.style,
            request_id=result["request_id"],
            moderated=result["moderated"],
            crafted_from_vault=True,
            vault_files_used=files_used,
        )

    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Vault video generation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/videos/status/{request_id}",
    response_model=VideoStatusResponse,
    summary="Check the status of a video generation request",
    tags=["Video Generation"],
)
async def videos_status(request_id: str):
    """
    Poll the status of a video generation or editing request by its `request_id`.

    Useful if you prefer to submit a job and check back later rather than
    waiting in the blocking `/videos/generate` or `/videos/edit` responses.
    You can get a `request_id` from the `request_id` field of any video
    response — it is always returned even when the request has already
    completed.

    **Status values:**
    - `pending` — still processing
    - `done` — complete; `video_url` is populated
    - `error` — generation failed

    This endpoint returns the *current* status without waiting — it does not
    block or retry.
    """
    try:
        resp = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: __import__("requests").get(
                f"https://api.x.ai/v1/videos/{request_id}",
                headers={"Authorization": f"Bearer {os.getenv('XAI_API_KEY', '')}"},
                timeout=15,
            ),
        )

        if not resp.ok:
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"xAI API error: {resp.text[:200]}",
            )

        data   = resp.json()
        status = data.get("status", "unknown")
        video  = data.get("video", {})

        return VideoStatusResponse(
            request_id=request_id,
            status=status,
            video_url=video.get("url") if video else None,
            duration=video.get("duration") if video else None,
            moderated=video.get("respect_moderation") if video else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Video status check failed")
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================================================
# fal.ai enhancement endpoints
# ===========================================================================

@app.post(
    "/enhance/upscale",
    response_model=ClarityUpscaleResponse,
    summary="Upscale an image using fal-ai/clarity-upscaler",
    tags=["Enhancement"],
)
async def enhance_upscale(request: ClarityUpscaleRequest):
    """
    Upscale and sharpen an image using **fal-ai/clarity-upscaler**.

    Unlike simple pixel scaling, Clarity uses a Stable Diffusion + ControlNet
    pipeline to reconstruct fine detail at higher resolution.  The result is
    stored on **fal.media** (persistent URL — no expiry), unlike the temporary
    URLs returned by Grok image generation.

    **Typical workflow:**
    1. Generate an image with `POST /images/generate` or `/images/from-vault`
    2. Pass the returned URL (even the temporary xAI URL) straight into this
       endpoint — it will be fetched and processed before it expires.
    3. Get back a permanent, high-resolution version.

    **Tuning `creativity` and `resemblance`:**

    | Content type | creativity | resemblance |
    |---|---|---|
    | Character portrait | 0.2–0.3 | 0.7–0.8 |
    | Environment / scene | 0.4–0.6 | 0.4–0.6 |
    | Map / diagram | 0.1 | 0.9 |

    Pass the original generation `prompt` for best results — Clarity uses it
    to guide what new detail gets added during the diffusion pass.
    """
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: clarity_upscale(
                image_url=request.image_url,
                upscale_factor=request.upscale_factor,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                creativity=request.creativity,
                resemblance=request.resemblance,
                guidance_scale=request.guidance_scale,
                num_inference_steps=request.num_inference_steps,
                seed=request.seed,
                enable_safety_checker=request.enable_safety_checker,
            ),
        )

        logger.info(
            f"Upscale complete | {result.get('width')}x{result.get('height')} "
            f"| {request.upscale_factor}x from {request.image_url[:50]}…"
        )

        return ClarityUpscaleResponse(
            image_url=result["image_url"],
            width=result.get("width"),
            height=result.get("height"),
            file_size=result.get("file_size"),
            content_type=result.get("content_type", "image/png"),
            seed=result.get("seed"),
            source_url=request.image_url,
            upscale_factor=request.upscale_factor,
        )

    except RuntimeError as e:
        # Missing FAL_KEY or fal-client not installed — surface clearly
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Clarity upscale failed")
        raise HTTPException(status_code=500, detail=str(e))
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
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agent import RPGAgent
from embeddings import VaultIndex
from models import (
    ChatRequest,
    CommitResponse,
    DiscardResponse,
    FileListResponse,
    FileSearchResult,
    IndexStatus,
    ModifyRequest,
    OperationRecord,
    PendingReview,
    StagedChangeResponse,
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

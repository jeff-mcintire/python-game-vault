"""
main.py — FastAPI application entry point.

Endpoints:
  POST /chat                       — Run agent (staged, nothing written yet)
  POST /review/{session_id}/confirm  — Commit staged changes to vault
  POST /review/{session_id}/modify   — Re-run agent with user feedback
  DELETE /review/{session_id}        — Discard staged changes

  GET  /status                     — Index health check
  GET  /files                      — List / search vault files
  POST /reindex                    — Trigger a full index rebuild

Workflow
────────
  1.  POST /chat  →  PendingReview   (session_id + list of StagedChanges with diffs)
  2a. POST /review/{id}/confirm      →  CommitResponse  (writes to disk)
  2b. POST /review/{id}/modify       →  PendingReview   (re-runs agent, returns new staged changes)
  2c. DELETE /review/{id}            →  DiscardResponse (drops the session, nothing written)
"""

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
    api_key = os.getenv("ANTHROPIC_API_KEY")
    cache_path = os.getenv("INDEX_CACHE_PATH", ".vault_index.pkl")

    if not vault_path:
        raise RuntimeError("VAULT_PATH environment variable is not set.")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")

    _vault = VaultManager(vault_path)
    _index = VaultIndex(cache_path=cache_path)
    _agent = RPGAgent(_vault, _index, api_key)

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

    logger.info("RPG Vault Agent is ready.")
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
    description="Claude-powered backend for managing your Obsidian RPG campaign vault.",
    version="2.0.0",
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
    """Convert a StagingArea into the PendingReview response model."""
    return PendingReview(
        session_id=staging.session_id,
        agent_response=staging.agent_response,
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


# ---------------------------------------------------------------------------
# POST /chat — Run the agent (staged)
# ---------------------------------------------------------------------------

@app.post(
    "/chat",
    response_model=PendingReview,
    summary="Run the agent and return proposed changes for review",
)
async def chat(request: ChatRequest):
    """
    Runs the agent against your prompt.  **Nothing is written to disk.**

    Returns a `PendingReview` containing:
    - `session_id`  — use this in subsequent `/review/*` calls
    - `changes`     — each proposed file operation with a unified diff
    - `agent_response` — Claude's plain-English summary of what it plans to do

    Next steps:
    - **Confirm** → `POST /review/{session_id}/confirm`
    - **Request changes** → `POST /review/{session_id}/modify`
    - **Discard** → `DELETE /review/{session_id}`
    """
    _require_agent()
    try:
        staging = StagingArea(original_prompt=request.prompt)
        result = _agent.run(request.prompt, top_k=request.top_k, staging=staging)

        staging.agent_response = result["response"]
        staging.files_referenced = result["files_referenced"]
        _sessions.put(staging)

        logger.info(
            f"Staged {len(staging.changes)} change(s) — session {staging.session_id}"
        )
        return _staging_to_response(staging, result["operations_performed"])

    except Exception as e:
        logger.exception("Agent error during staging")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# POST /review/{session_id}/confirm — Commit to disk
# ---------------------------------------------------------------------------

@app.post(
    "/review/{session_id}/confirm",
    response_model=CommitResponse,
    summary="Commit all staged changes to the vault",
)
async def confirm(session_id: str):
    """
    Writes all staged changes to the vault and cleans up the session.

    After this call the files are on disk and the index is updated.
    """
    _require_agent()
    staging = _require_session(session_id)

    if not staging.has_changes:
        _sessions.discard(session_id)
        return CommitResponse(
            session_id=session_id,
            files_committed=[],
            message="No changes to commit.",
        )

    try:
        committed = staging.commit(_vault)

        # Update the embedding index for each committed file
        for rel_path in committed:
            try:
                full_text = _vault.read_relative(rel_path)
                _index.update_file(rel_path, full_text)
            except Exception:
                pass  # delete ops won't be readable; that's fine

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
# POST /review/{session_id}/modify — Re-run with feedback
# ---------------------------------------------------------------------------

@app.post(
    "/review/{session_id}/modify",
    response_model=PendingReview,
    summary="Re-run the agent with modification feedback",
)
async def modify(session_id: str, request: ModifyRequest):
    """
    Discards the current staged changes and re-runs the agent with your
    original prompt **plus** your modification feedback.

    Returns a fresh `PendingReview` with a new `session_id`.
    You can confirm, modify again, or discard as normal.

    Example feedback:
    - *"Give the blacksmith a backstory as a former soldier"*
    - *"Use the Eastwall district instead of Stormhaven"*
    - *"Add a rival NPC who runs a competing shop"*
    """
    _require_agent()
    staging = _require_session(session_id)

    # Build a combined prompt: original request + user feedback
    combined_prompt = (
        f"{staging.original_prompt}\n\n"
        f"---\n\n"
        f"**Revision request:** {request.feedback}"
    )

    try:
        # Discard old session
        _sessions.discard(session_id)

        # Fresh staging area with the combined prompt
        new_staging = StagingArea(original_prompt=combined_prompt)
        result = _agent.run(combined_prompt, staging=new_staging)

        new_staging.agent_response = result["response"]
        new_staging.files_referenced = result["files_referenced"]
        _sessions.put(new_staging)

        logger.info(
            f"Modify re-run → {len(new_staging.changes)} staged change(s) "
            f"— new session {new_staging.session_id}"
        )
        return _staging_to_response(new_staging, result["operations_performed"])

    except Exception as e:
        logger.exception("Agent error during modify")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# DELETE /review/{session_id} — Discard
# ---------------------------------------------------------------------------

@app.delete(
    "/review/{session_id}",
    response_model=DiscardResponse,
    summary="Discard staged changes without writing anything",
)
async def discard(session_id: str):
    """
    Drops the staging session.  Nothing is written to disk.
    """
    found = _sessions.discard(session_id)
    if not found:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found or already discarded.",
        )
    logger.info(f"Discarded session {session_id}")
    return DiscardResponse(
        session_id=session_id,
        message="Staged changes discarded. Nothing was written to the vault.",
    )


# ---------------------------------------------------------------------------
# GET /status
# ---------------------------------------------------------------------------

@app.get("/status", response_model=IndexStatus, summary="Index + watcher health")
async def status():
    if _vault is None:
        raise HTTPException(status_code=503, detail="Not initialised.")
    _status["indexed_files"] = _index.file_count
    _status["total_files"] = len(_vault.scan_files())
    return IndexStatus(vault_path=str(_vault.vault_path), **_status)


# ---------------------------------------------------------------------------
# GET /files
# ---------------------------------------------------------------------------

@app.get("/files", response_model=FileListResponse, summary="List or search vault files")
async def list_files(query: str = ""):
    if _index is None:
        raise HTTPException(status_code=503, detail="Not initialised.")
    if query:
        results = _index.search(query, top_k=20)
        return FileListResponse(
            files=[FileSearchResult(path=p, score=s) for p, s in results]
        )
    return FileListResponse(
        files=[FileSearchResult(path=p, score=1.0) for p in _index.file_paths]
    )


# ---------------------------------------------------------------------------
# POST /reindex
# ---------------------------------------------------------------------------

@app.post("/reindex", summary="Trigger a full index rebuild")
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

"""
main.py — FastAPI application entry point.

Endpoints:
  POST /chat          — Main agent endpoint
  GET  /status        — Index health check
  GET  /files         — List / search vault files
  POST /reindex       — Trigger a full rebuild in the background
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
    ChatResponse,
    FileListResponse,
    FileSearchResult,
    IndexStatus,
    OperationRecord,
)
from vault import VaultManager
from watcher import start_watcher

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application state (populated during startup)
# ---------------------------------------------------------------------------

_vault: VaultManager | None = None
_index: VaultIndex | None = None
_agent: RPGAgent | None = None
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
# Lifespan (startup / shutdown)
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

    # --- initialise core objects ---
    _vault = VaultManager(vault_path)
    _index = VaultIndex(cache_path=cache_path)
    _agent = RPGAgent(_vault, _index, api_key)

    # --- load or build index ---
    if not _index.load_cache():
        logger.info("No cache found — building embedding index from scratch…")
        files = _vault.scan_files()
        file_data = [_vault.read_file(f) for f in files]
        _index.build_index(file_data)
        _status["last_indexed"] = _now_iso()
    else:
        _status["last_indexed"] = _now_iso()  # cache was there

    _status["total_files"] = len(_vault.scan_files())
    _status["indexed_files"] = _index.file_count

    # --- start filesystem watcher ---
    _watcher = start_watcher(_vault, _index)
    _status["watching"] = True

    logger.info("RPG Vault Agent is ready.")
    yield

    # --- shutdown ---
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
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse, summary="Send a prompt to the agent")
async def chat(request: ChatRequest):
    """
    Send a natural-language prompt.  The agent will:
    1. Semantic-search the vault for relevant files.
    2. Pass context + prompt to Claude.
    3. Execute any tool calls (create / update / append files).
    4. Return a summary + list of affected files.
    """
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised.")
    try:
        result = _agent.run(request.prompt, top_k=request.top_k)
        return ChatResponse(
            response=result["response"],
            files_referenced=result["files_referenced"],
            files_modified=result["files_modified"],
            operations_performed=[
                OperationRecord(**op) for op in result["operations_performed"]
            ],
        )
    except Exception as e:
        logger.exception("Agent error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status", response_model=IndexStatus, summary="Index + watcher health")
async def status():
    if _vault is None:
        raise HTTPException(status_code=503, detail="Not initialised.")
    _status["indexed_files"] = _index.file_count
    _status["total_files"] = len(_vault.scan_files())
    return IndexStatus(vault_path=str(_vault.vault_path), **_status)


@app.get("/files", response_model=FileListResponse, summary="List or search vault files")
async def list_files(query: str = ""):
    """
    Without `query`: returns all indexed file paths.
    With `query`: returns the top 20 most semantically relevant files.
    """
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


@app.post("/reindex", summary="Trigger a full index rebuild")
async def reindex(background_tasks: BackgroundTasks):
    """
    Drops the current index and rebuilds it from scratch.
    Runs in the background so the request returns immediately.
    Useful after bulk edits to the vault outside of this app.
    """
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

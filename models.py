from typing import Literal, Optional
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class OperationRecord(BaseModel):
    operation: str          # create | update | append | delete | read | error
    path: Optional[str] = None
    tool: Optional[str] = None


# ---------------------------------------------------------------------------
# Chat  (POST /chat)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    prompt: str
    top_k: int = 10


# ---------------------------------------------------------------------------
# Staged change  (returned inside PendingReview)
# ---------------------------------------------------------------------------

class StagedChangeResponse(BaseModel):
    operation: Literal["create", "update", "append", "delete"]
    relative_path: str
    proposed_content: str
    original_content: Optional[str] = None   # None for new files
    diff: str


# ---------------------------------------------------------------------------
# Pending review  (returned by POST /chat)
# ---------------------------------------------------------------------------

class PendingReview(BaseModel):
    """
    Returned when the agent has finished planning but nothing has been
    written to disk yet.  The client should display the staged changes
    and let the user confirm, modify, or discard.
    """
    session_id: str
    agent_response: str                       # Claude's plain-English summary
    files_referenced: list[str]
    changes: list[StagedChangeResponse]       # all proposed file operations
    operations_performed: list[OperationRecord]


# ---------------------------------------------------------------------------
# Review actions
# ---------------------------------------------------------------------------

class ModifyRequest(BaseModel):
    feedback: str    # user's modification instructions


class CommitResponse(BaseModel):
    session_id: str
    files_committed: list[str]
    message: str


class DiscardResponse(BaseModel):
    session_id: str
    message: str


# ---------------------------------------------------------------------------
# Other endpoints
# ---------------------------------------------------------------------------

class IndexStatus(BaseModel):
    vault_path: str
    total_files: int
    indexed_files: int
    last_indexed: Optional[str] = None
    watching: bool = False


class FileSearchResult(BaseModel):
    path: str
    score: float


class FileListResponse(BaseModel):
    files: list[FileSearchResult]

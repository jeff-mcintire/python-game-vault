from typing import Literal, Optional
from pydantic import BaseModel
from providers import ProviderName


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
    provider: ProviderName = ProviderName.CLAUDE   # "claude" or "grok"
    model: Optional[str] = None                    # override default model for chosen provider


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
    agent_response: str
    provider: ProviderName                        # which provider was used
    model: str                                    # which model was used
    files_referenced: list[str]
    changes: list[StagedChangeResponse]
    operations_performed: list[OperationRecord]


# ---------------------------------------------------------------------------
# Review actions
# ---------------------------------------------------------------------------

class ModifyRequest(BaseModel):
    feedback: str
    provider: Optional[ProviderName] = None   # optionally switch provider for the re-run
    model: Optional[str] = None               # optionally switch model for the re-run


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


# ---------------------------------------------------------------------------
# Image generation  (POST /images/generate  and  POST /images/from-vault)
# ---------------------------------------------------------------------------

class ImageGenerateRequest(BaseModel):
    prompt: str                          # raw image prompt
    n: int = 2                           # number of images to return (default 2)


class VaultImageRequest(BaseModel):
    description: str                     # what you want to visualize
    vault_references: list[str] = []     # optional explicit file paths to include
    top_k: int = 6                       # how many semantic search results to pull
    n: int = 2                           # number of images to return


class ImageGenerateResponse(BaseModel):
    images: list[str]                    # list of image URLs
    prompt_used: str                     # the prompt that was sent to the image model
    crafted_from_vault: bool = False     # True when the prompt was built from vault content
    vault_files_used: list[str] = []     # which vault files informed the prompt

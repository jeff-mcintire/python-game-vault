from typing import Literal, Optional
from pydantic import BaseModel, field_validator
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

# Importing valid option lists from image_gen keeps models and logic in sync.
from image_gen import ASPECT_RATIOS, RESOLUTIONS, STYLES


class ImageGenerateRequest(BaseModel):
    prompt: str
    n: int = 2
    # --- display options ---
    aspect_ratio: str = "auto"
    """
    Image proportions.  One of:
    auto | 1:1 | 16:9 | 9:16 | 4:3 | 3:4 | 3:2 | 2:3 | 2:1 | 1:2
    19.5:9 | 9:19.5 | 20:9 | 9:20
    Default: auto (model picks the best ratio for the prompt).
    """
    resolution: str = "1k"
    """Output resolution.  One of: 1k | 2k.  Default: 1k."""
    style: Optional[str] = None
    """
    Art style preset appended to the prompt.  One of:
    photorealistic | oil_painting | watercolor | pencil_sketch |
    anime | dark_fantasy | concept_art | ink_wash
    Default: None (no style directive added).
    """

    @field_validator("aspect_ratio")
    @classmethod
    def validate_aspect_ratio(cls, v: str) -> str:
        if v not in ASPECT_RATIOS:
            raise ValueError(f"aspect_ratio must be one of {ASPECT_RATIOS}")
        return v

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v: str) -> str:
        if v not in RESOLUTIONS:
            raise ValueError(f"resolution must be one of {RESOLUTIONS}")
        return v

    @field_validator("style")
    @classmethod
    def validate_style(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in STYLES:
            raise ValueError(f"style must be one of {STYLES} or null")
        return v


class VaultImageRequest(BaseModel):
    description: str                     # what you want to visualize
    vault_references: list[str] = []     # optional explicit file paths to include
    top_k: int = 6                       # how many semantic search results to pull
    n: int = 2
    # --- display options (same as ImageGenerateRequest) ---
    aspect_ratio: str = "auto"
    resolution: str = "1k"
    style: Optional[str] = None

    @field_validator("aspect_ratio")
    @classmethod
    def validate_aspect_ratio(cls, v: str) -> str:
        if v not in ASPECT_RATIOS:
            raise ValueError(f"aspect_ratio must be one of {ASPECT_RATIOS}")
        return v

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v: str) -> str:
        if v not in RESOLUTIONS:
            raise ValueError(f"resolution must be one of {RESOLUTIONS}")
        return v

    @field_validator("style")
    @classmethod
    def validate_style(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in STYLES:
            raise ValueError(f"style must be one of {STYLES} or null")
        return v


class ImageGenerateResponse(BaseModel):
    images: list[str]                    # list of image URLs
    prompt_used: str                     # the final prompt sent to the image model
    aspect_ratio: str = "auto"           # aspect ratio used
    resolution: str = "1k"              # resolution used
    style: Optional[str] = None          # style preset used (if any)
    crafted_from_vault: bool = False
    vault_files_used: list[str] = []
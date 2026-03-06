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

# Image provider choices
IMAGE_PROVIDERS = ["aurora", "flux2pro"]
"""
aurora   — Grok Aurora (xAI).  Photorealistic, cinematic.
           Requires XAI_API_KEY.  URLs are temporary.
flux2pro — FLUX 2 Pro (Black Forest Labs via fal.ai).  Illustrated /
           painterly / stylised.  Commercially licensed.
           Requires FAL_KEY.  URLs are persistent.
"""

FLUX_SAFETY_TOLERANCES = ["1", "2", "3", "4", "5", "6"]


class ImageGenerateRequest(BaseModel):
    prompt: str
    n: int = 2
    # --- provider ---
    provider: str = "aurora"
    """
    Image generation backend.  One of: aurora | flux2pro
      aurora   — Grok Aurora (xAI).  Photorealistic, cinematic.
      flux2pro — FLUX 2 Pro (fal.ai).  Illustrated / painterly / stylised.
    Default: aurora.
    """
    # --- display options ---
    aspect_ratio: str = "auto"
    """
    Image proportions.  One of:
    auto | 1:1 | 16:9 | 9:16 | 4:3 | 3:4 | 3:2 | 2:3 | 2:1 | 1:2
    19.5:9 | 9:19.5 | 20:9 | 9:20
    Default: auto (model picks the best ratio for the prompt).
    Note: FLUX 2 Pro maps these to its nearest preset automatically.
    """
    resolution: str = "1k"
    """Output resolution.  One of: 1k | 2k.  Default: 1k.  (Aurora only — ignored for FLUX 2 Pro.)"""
    style: Optional[str] = None
    """
    Art style preset appended to the prompt.  One of:
    photorealistic | oil_painting | watercolor | pencil_sketch |
    anime | dark_fantasy | concept_art | ink_wash
    Default: None (no style directive added).
    """
    # --- FLUX 2 Pro options (ignored when provider = aurora) ---
    seed: Optional[int] = None
    """Seed for reproducible output.  FLUX 2 Pro only."""
    safety_tolerance: str = "2"
    """
    Safety filter strictness.  One of: 1 | 2 | 3 | 4 | 5 | 6
    1 = most strict, 6 = most permissive.  Default: 2.
    Use 5 for dark fantasy / mature RPG content.  FLUX 2 Pro only.
    """
    enable_safety_checker: bool = True
    """
    Whether to run the built-in safety checker.  Default: True.
    Set to false for dark fantasy / mature RPG content.  FLUX 2 Pro only.
    Note: safety_tolerance is the finer-grained dial; disabling the checker
    entirely removes the gate altogether.
    """

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        if v not in IMAGE_PROVIDERS:
            raise ValueError(f"provider must be one of {IMAGE_PROVIDERS}")
        return v

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

    @field_validator("safety_tolerance")
    @classmethod
    def validate_safety_tolerance(cls, v: str) -> str:
        if v not in FLUX_SAFETY_TOLERANCES:
            raise ValueError(f"safety_tolerance must be one of {FLUX_SAFETY_TOLERANCES}")
        return v


class VaultImageRequest(BaseModel):
    description: str                     # what you want to visualize
    vault_references: list[str] = []     # optional explicit file paths to include
    top_k: int = 6                       # how many semantic search results to pull
    n: int = 2
    # --- provider ---
    provider: str = "aurora"
    """Image generation backend.  One of: aurora | flux2pro  (default: aurora)."""
    # --- display options (same as ImageGenerateRequest) ---
    aspect_ratio: str = "auto"
    resolution: str = "1k"
    style: Optional[str] = None
    # --- FLUX 2 Pro options ---
    seed: Optional[int] = None
    safety_tolerance: str = "2"
    enable_safety_checker: bool = True
    """Whether to run the built-in safety checker.  Default: True.  FLUX 2 Pro only."""

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        if v not in IMAGE_PROVIDERS:
            raise ValueError(f"provider must be one of {IMAGE_PROVIDERS}")
        return v

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

    @field_validator("safety_tolerance")
    @classmethod
    def validate_safety_tolerance(cls, v: str) -> str:
        if v not in FLUX_SAFETY_TOLERANCES:
            raise ValueError(f"safety_tolerance must be one of {FLUX_SAFETY_TOLERANCES}")
        return v


class ImageGenerateResponse(BaseModel):
    images: list[str]                    # list of image URLs
    prompt_used: str                     # the final prompt sent to the image model
    provider: str = "aurora"             # which image backend was used
    aspect_ratio: str = "auto"           # aspect ratio used
    resolution: str = "1k"              # resolution used (aurora only)
    style: Optional[str] = None          # style preset used (if any)
    crafted_from_vault: bool = False
    vault_files_used: list[str] = []


# ---------------------------------------------------------------------------
# Video generation  (POST /videos/generate, /videos/edit, /videos/from-vault)
# ---------------------------------------------------------------------------

from video_gen import VIDEO_ASPECT_RATIOS, VIDEO_RESOLUTIONS


class VideoGenerateRequest(BaseModel):
    """
    Direct text-to-video (or image-to-video) request.

    Set image_url to animate a still image instead of generating from text.
    """
    prompt: str
    duration: int = 8
    """
    Video length in seconds.  Range: 1–15.  Default: 8.
    Ignored when editing an existing video (the output matches the input length).
    """
    aspect_ratio: str = "16:9"
    """
    Video proportions.  One of: 16:9 | 9:16 | 4:3 | 3:4 | 1:1.
    Default: 16:9 (landscape widescreen).
    """
    resolution: str = "720p"
    """Output resolution.  One of: 480p | 720p | 1080p.  Default: 720p."""
    image_url: Optional[str] = None
    """
    Optional.  Provide a publicly accessible image URL to use image-to-video
    mode — the model will animate this image guided by your prompt.
    """
    style: Optional[str] = None
    """
    Optional free-text style direction woven into the prompt before submission.
    Unlike image generation, video style is not a preset list — describe it
    freely: "slow-motion", "handheld documentary", "sweeping aerial shot", etc.
    """

    @field_validator("duration")
    @classmethod
    def validate_duration(cls, v: int) -> int:
        if not (1 <= v <= 15):
            raise ValueError("duration must be between 1 and 15 seconds")
        return v

    @field_validator("aspect_ratio")
    @classmethod
    def validate_aspect_ratio(cls, v: str) -> str:
        if v not in VIDEO_ASPECT_RATIOS:
            raise ValueError(f"aspect_ratio must be one of {VIDEO_ASPECT_RATIOS}")
        return v

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v: str) -> str:
        if v not in VIDEO_RESOLUTIONS:
            raise ValueError(f"resolution must be one of {VIDEO_RESOLUTIONS}")
        return v


class VideoEditRequest(BaseModel):
    """
    Edit an existing video by describing the desired changes.
    Input video must be ≤ 8.7 seconds.  Output duration matches the input.
    """
    video_url: str
    """Publicly accessible URL to the source video (max 8.7 s)."""
    prompt: str
    """Describe what should change: "make the sky stormy", "remove the guards", etc."""


class VaultVideoRequest(BaseModel):
    """
    Vault-aware video generation.  Describe what you want to see and
    optionally name specific vault files; Grok will craft a lore-accurate
    cinematic prompt before generating.
    """
    description: str
    vault_references: list[str] = []
    """Explicit vault file paths to load (e.g. ["NPCs/Sable.md"])."""
    top_k: int = 6
    """Number of additional files to pull via semantic search."""
    # --- video parameters ---
    duration: int = 8
    aspect_ratio: str = "16:9"
    resolution: str = "720p"
    style: Optional[str] = None
    """
    Free-text style direction for the vault prompt builder.
    Examples: "epic slow push-in", "handheld night scene", "dark fantasy aerial".
    """

    @field_validator("duration")
    @classmethod
    def validate_duration(cls, v: int) -> int:
        if not (1 <= v <= 15):
            raise ValueError("duration must be between 1 and 15 seconds")
        return v

    @field_validator("aspect_ratio")
    @classmethod
    def validate_aspect_ratio(cls, v: str) -> str:
        if v not in VIDEO_ASPECT_RATIOS:
            raise ValueError(f"aspect_ratio must be one of {VIDEO_ASPECT_RATIOS}")
        return v

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v: str) -> str:
        if v not in VIDEO_RESOLUTIONS:
            raise ValueError(f"resolution must be one of {VIDEO_RESOLUTIONS}")
        return v


class VideoGenerateResponse(BaseModel):
    video_url: str                       # URL of the generated/edited video
    prompt_used: str                     # final prompt sent to the model
    duration: int                        # actual video length in seconds
    aspect_ratio: str                    # aspect ratio used
    resolution: str                      # resolution used
    style: Optional[str] = None          # style direction (if any)
    request_id: str                      # xAI request ID (for your records)
    moderated: bool = True               # whether the video passed content moderation
    crafted_from_vault: bool = False
    vault_files_used: list[str] = []


class VideoStatusResponse(BaseModel):
    request_id: str
    status: str                          # "pending" | "done" | "error"
    video_url: Optional[str] = None
    duration: Optional[int] = None
    moderated: Optional[bool] = None


# ---------------------------------------------------------------------------
# fal.ai tools  (POST /enhance/upscale)
# ---------------------------------------------------------------------------

class ClarityUpscaleRequest(BaseModel):
    """
    Upscale an image using fal-ai/clarity-upscaler.

    Supply any publicly accessible image URL — including the temporary URLs
    returned by the Grok image generation endpoints.  The output is stored
    on fal.media and does not expire.
    """
    image_url: str
    """
    Public URL or base64 data URI of the image to upscale.
    Accepts xAI temporary URLs, fal.media URLs, or any direct image link.
    """

    upscale_factor: float = 2.0
    """
    Scale multiplier.  2 doubles the resolution; 4 quadruples it.
    Default: 2.
    """

    prompt: str = "masterpiece, best quality, highres"
    """
    Steers what detail gets added during the diffusion upscale pass.
    Pass the original image generation prompt for best results.
    Default: "masterpiece, best quality, highres"
    """

    negative_prompt: str = "(worst quality, low quality, normal quality:2)"
    """What to avoid in the added detail."""

    creativity: float = 0.35
    """
    0–1.  How much the model deviates from the original image.
    Low (0.1–0.3) = faithful, minimal invention.
    High (0.5–0.8) = more invented texture and atmosphere.

    RPG guide:
      Character portraits  → 0.2–0.3
      Environments/scenes  → 0.4–0.6
      Maps / diagrams      → 0.1
    Default: 0.35
    """

    resemblance: float = 0.6
    """
    0–1.  ControlNet strength — how closely the output tracks the original
    structure.  Higher = closer to original.  Default: 0.6
    """

    guidance_scale: float = 4.0
    """CFG scale — how strictly the model follows the prompt.  Default: 4."""

    num_inference_steps: int = 18
    """
    Inference steps.  More steps = higher quality, slower processing.
    Range: 10–50.  Default: 18 (good balance of speed and quality).
    """

    seed: Optional[int] = None
    """Set for reproducible output.  Omit for a random seed."""

    enable_safety_checker: bool = True
    """
    Set to false to disable the safety checker.
    Recommended false for dark fantasy / mature RPG content.
    """

    @field_validator("creativity", "resemblance")
    @classmethod
    def validate_zero_to_one(cls, v: float, info) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{info.field_name} must be between 0.0 and 1.0")
        return v

    @field_validator("upscale_factor")
    @classmethod
    def validate_upscale_factor(cls, v: float) -> float:
        if not (1.0 < v <= 8.0):
            raise ValueError("upscale_factor must be between 1.0 and 8.0")
        return v

    @field_validator("num_inference_steps")
    @classmethod
    def validate_steps(cls, v: int) -> int:
        if not (1 <= v <= 100):
            raise ValueError("num_inference_steps must be between 1 and 100")
        return v


class ClarityUpscaleResponse(BaseModel):
    image_url: str          # fal.media-hosted output URL (persistent)
    width: Optional[int]    # output width in pixels
    height: Optional[int]   # output height in pixels
    file_size: Optional[int] # output file size in bytes
    content_type: str = "image/png"
    seed: Optional[int]     # seed used for reproducibility
    source_url: str         # the original image that was upscaled
    upscale_factor: float   # scale factor used
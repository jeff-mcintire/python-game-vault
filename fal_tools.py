"""
fal_tools.py — fal.ai model integrations.

fal.ai is a serverless GPU inference platform hosting 600+ models.
All fal.ai models share the same authentication (FAL_KEY env var) and the
same call pattern:

    result = fal_client.run("model-id/name", arguments={...})

This module provides wrappers for specific fal.ai tools.  Adding a new tool
later means adding one function here, one Pydantic model in models.py, and
one endpoint in main.py.

Authentication: set FAL_KEY in .env
Install:        uv add fal-client   (or: uv sync after adding to pyproject.toml)

---

Implemented tools
-----------------

  clarity_upscale(image_url, ...)
    fal-ai/clarity-upscaler
    Diffusion-based upscaler — intelligently adds detail rather than just
    scaling pixels.  Uses a Stable Diffusion pipeline with ControlNet to
    stay close to the original while generating sharper, higher-resolution
    output.

    Key tuning parameters:
      upscale_factor  — how much to enlarge (default 2x)
      creativity      — how much new detail to invent (0–1, default 0.35)
                        low  = faithful to original
                        high = more invented texture / atmosphere
      resemblance     — ControlNet strength; how closely to track the source
                        (0–1, default 0.6)
      prompt          — steers what detail gets added during upscaling;
                        pass the original generation prompt for best results

    Recommended settings for RPG art:
      Character portraits  → creativity 0.2–0.3, resemblance 0.7–0.8
      Environments/scenes  → creativity 0.4–0.6, resemblance 0.4–0.6
      Maps / diagrams      → creativity 0.1, resemblance 0.9

"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

FAL_CLARITY_MODEL = "fal-ai/clarity-upscaler"


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _get_fal_client():
    """
    Import and configure fal_client with FAL_KEY.
    Lazy import so the module loads even if fal-client is not installed yet,
    and raises a clear error only when a fal tool is actually called.
    """
    try:
        import fal_client
    except ImportError:
        raise RuntimeError(
            "fal-client is not installed.  "
            "Run: uv add fal-client  (or uv sync after updating pyproject.toml)"
        )

    key = os.getenv("FAL_KEY")
    if not key:
        raise RuntimeError(
            "FAL_KEY is not set.  Add it to your .env file."
        )

    os.environ["FAL_KEY"] = key  # fal_client reads from env
    return fal_client


# ---------------------------------------------------------------------------
# Clarity Upscaler
# ---------------------------------------------------------------------------

def clarity_upscale(
    image_url: str,
    upscale_factor: float = 2.0,
    prompt: str = "masterpiece, best quality, highres",
    negative_prompt: str = "(worst quality, low quality, normal quality:2)",
    creativity: float = 0.35,
    resemblance: float = 0.6,
    guidance_scale: float = 4.0,
    num_inference_steps: int = 18,
    seed: Optional[int] = None,
    enable_safety_checker: bool = True,
) -> dict:
    """
    Upscale an image using fal-ai/clarity-upscaler.

    Unlike a simple resize, Clarity uses a Stable Diffusion pipeline to
    reconstruct fine detail at higher resolution.  The image is first
    enlarged, then ControlNet steers the diffusion pass to stay close to
    the original while adding sharpness and texture.

    Parameters
    ----------
    image_url           Public URL or base64 data URI of the source image.
    upscale_factor      Scale multiplier (2 = 2x resolution).  Default: 2.
    prompt              Prompt that steers what detail gets added.  Passing
                        the original generation prompt gives the best results.
    negative_prompt     What to avoid in the added detail.
    creativity          0–1.  How much the model deviates from the source.
                        Low = faithful; high = more invented texture.
    resemblance         0–1.  ControlNet strength.  How closely the output
                        tracks the original structure.
    guidance_scale      CFG scale.  How strictly the model follows the prompt.
    num_inference_steps Inference steps — higher = more quality, slower.
    seed                Set for reproducible output.  None = random.
    enable_safety_checker  Disable for dark fantasy / mature content.

    Returns
    -------
    dict with keys:
        image_url   str   — fal.media-hosted output URL (persistent)
        width       int   — output width in pixels
        height      int   — output height in pixels
        file_size   int   — output file size in bytes
        seed        int   — seed used
        content_type str  — e.g. "image/png"
    """
    fal_client = _get_fal_client()

    arguments = {
        "image_url":             image_url,
        "prompt":                prompt,
        "upscale_factor":        upscale_factor,
        "negative_prompt":       negative_prompt,
        "creativity":            creativity,
        "resemblance":           resemblance,
        "guidance_scale":        guidance_scale,
        "num_inference_steps":   num_inference_steps,
        "enable_safety_checker": enable_safety_checker,
    }
    if seed is not None:
        arguments["seed"] = seed

    logger.info(
        f"Clarity upscale | factor={upscale_factor}x "
        f"creativity={creativity} resemblance={resemblance} "
        f"steps={num_inference_steps} | {image_url[:60]}…"
    )

    result = fal_client.run(FAL_CLARITY_MODEL, arguments=arguments)

    image = result.get("image", {})
    output = {
        "image_url":    image.get("url", ""),
        "width":        image.get("width"),
        "height":       image.get("height"),
        "file_size":    image.get("file_size"),
        "content_type": image.get("content_type", "image/png"),
        "seed":         result.get("seed"),
    }

    logger.info(
        f"Clarity upscale done | "
        f"{output['width']}x{output['height']} "
        f"({(output['file_size'] or 0) // 1024} KB) | {output['image_url'][:60]}…"
    )
    return output

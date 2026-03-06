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

  flux2_pro_generate(prompt, n, image_size, seed, safety_tolerance, ...)
    fal-ai/flux-2-pro  (Black Forest Labs, commercially licensed)
    Zero-config text-to-image generation — 32B parameter model, production
    optimised, no inference steps or guidance scales to configure.  Excellent
    for illustrated/painterly RPG art; a stylistic complement to Grok Aurora's
    photorealism.

    Aspect ratio is controlled via `image_size` (FLUX's enum, not "16:9" style).
    Use the convenience wrapper flux2_pro_generate() which accepts the same
    "16:9" / "1:1" strings as the Grok endpoints and maps them automatically.

    safety_tolerance: "1" (strict) → "6" (permissive).  Set to "5" or "6"
    for dark fantasy / mature content (disabling the checker outright is not
    recommended; tolerance "5" handles most RPG art needs).

    Output URLs are persistent fal.media links (unlike xAI's temporary URLs).

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

FAL_CLARITY_MODEL  = "fal-ai/clarity-upscaler"
FAL_FLUX2_PRO_MODEL = "fal-ai/flux-2-pro"

# ---------------------------------------------------------------------------
# Aspect-ratio translation: "16:9" style → FLUX image_size enum
# ---------------------------------------------------------------------------
# FLUX 2 Pro uses named size presets, not ratio strings.
# We map the shared ratio vocabulary to the closest FLUX preset.

_FLUX_SIZE_MAP: dict[str, str] = {
    "auto":    "landscape_4_3",   # sensible cinematic default
    "1:1":     "square_hd",
    "16:9":    "landscape_16_9",
    "9:16":    "portrait_16_9",
    "4:3":     "landscape_4_3",
    "3:4":     "portrait_4_3",
    "3:2":     "landscape_4_3",   # closest preset
    "2:3":     "portrait_4_3",    # closest preset
    "2:1":     "landscape_16_9",  # widest landscape available
    "1:2":     "portrait_16_9",   # tallest portrait available
    "19.5:9":  "landscape_16_9",
    "9:19.5":  "portrait_16_9",
    "20:9":    "landscape_16_9",
    "9:20":    "portrait_16_9",
}

FLUX2_PRO_IMAGE_SIZES = [
    "square_hd", "square",
    "portrait_4_3", "portrait_16_9",
    "landscape_4_3", "landscape_16_9",
]


def _aspect_to_flux_size(aspect_ratio: str) -> str:
    """Convert a shared aspect-ratio string to a FLUX image_size enum value."""
    return _FLUX_SIZE_MAP.get(aspect_ratio, "landscape_4_3")


# ---------------------------------------------------------------------------
# FLUX 2 Pro — text-to-image generation
# ---------------------------------------------------------------------------

def flux2_pro_generate(
    prompt: str,
    n: int = 1,
    aspect_ratio: str = "auto",
    seed: Optional[int] = None,
    safety_tolerance: str = "2",
    enable_safety_checker: bool = True,
    output_format: str = "jpeg",
) -> list[str]:
    """
    Generate images using fal-ai/flux-2-pro (Black Forest Labs, commercially
    licensed).

    FLUX 2 Pro is a zero-configuration 32B text-to-image model optimised for
    production workflows.  It excels at illustrated/painterly and stylised RPG
    art — a strong complement to Grok Aurora's photorealism.

    Unlike Aurora (which accepts `n` natively), FLUX 2 Pro generates one image
    per API call.  When n > 1 this function fires n sequential calls and
    collects the results.

    Parameters
    ----------
    prompt              Text description of the image to generate.
    n                   Number of images to generate (default 1).
    aspect_ratio        Shared ratio string ("16:9", "1:1", etc.).  Mapped
                        automatically to the FLUX image_size enum.
    seed                Optional seed for reproducibility.
    safety_tolerance    "1" (strict) → "6" (permissive).  Use "5" for dark
                        fantasy / mature content.  Default: "2".
    enable_safety_checker  Whether to enable the built-in safety checker.
                        Default: True.
    output_format       "jpeg" (default) or "png".

    Returns
    -------
    list[str]   Persistent fal.media-hosted image URLs.
    """
    fal_client = _get_fal_client()

    image_size = _aspect_to_flux_size(aspect_ratio)

    arguments: dict = {
        "prompt":                 prompt,
        "image_size":             image_size,
        "safety_tolerance":       safety_tolerance,
        "enable_safety_checker":  enable_safety_checker,
        "output_format":          output_format,
    }
    if seed is not None:
        arguments["seed"] = seed

    logger.info(
        f"FLUX 2 Pro generate | n={n} size={image_size} "
        f"tolerance={safety_tolerance} | prompt: {prompt[:80]}…"
    )

    urls: list[str] = []
    for i in range(n):
        result = fal_client.run(FAL_FLUX2_PRO_MODEL, arguments=arguments)
        images = result.get("images", [])
        for img in images:
            url = img.get("url", "")
            if url:
                urls.append(url)
        logger.info(f"  Image {i + 1}/{n}: {(urls[-1] if urls else 'no url')[:60]}…")

    logger.info(f"FLUX 2 Pro done | {len(urls)} image(s) generated")
    return urls


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
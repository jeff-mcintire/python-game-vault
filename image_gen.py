"""
image_gen.py — xAI Grok image generation.

Two modes:

  generate_images(prompt, n, aspect_ratio, resolution, style)
    Direct generation — takes a prompt string plus optional display settings,
    returns n image URLs.
    Uses POST https://api.x.ai/v1/images/generations via the OpenAI SDK.

  build_vault_prompt(description, vault_context, style, api_key)
    Vault-aware prompt builder — feeds the user's description plus relevant
    vault file content to Grok chat, which crafts a rich, detailed image
    generation prompt grounded in the campaign lore.
    Returns the crafted prompt string.

Supported API parameters (as of grok-imagine-image):
  aspect_ratio  — "1:1" | "16:9" | "9:16" | "4:3" | "3:4" | "3:2" | "2:3"
                  "2:1" | "1:2" | "19.5:9" | "9:19.5" | "20:9" | "9:20" | "auto"
  resolution    — "1k" | "2k"

Note: "style" is NOT a native API param.  It is appended to the prompt as a
      text instruction (e.g. "rendered as an oil painting").  The xAI API does
      not accept quality/size/style fields — those are OpenAI-specific and will
      cause a 400 error if sent.

Model: grok-imagine-image
"""

import logging
import os
from typing import Literal, Optional

import requests
from openai import OpenAI

logger = logging.getLogger(__name__)

IMAGE_MODEL   = "grok-imagine-image"
CHAT_MODEL    = "grok-4-1-fast-reasoning"
GROK_BASE_URL = "https://api.x.ai/v1"

# ---------------------------------------------------------------------------
# Valid values — used for validation in models.py and docs
# ---------------------------------------------------------------------------

ASPECT_RATIOS = [
    "auto",
    "1:1",
    "16:9", "9:16",
    "4:3",  "3:4",
    "3:2",  "2:3",
    "2:1",  "1:2",
    "19.5:9", "9:19.5",
    "20:9", "9:20",
]

RESOLUTIONS = ["1k", "2k"]

# Style presets — appended as prompt text, not sent as API params
STYLE_SUFFIXES: dict[str, str] = {
    "photorealistic":  "ultra-photorealistic photography, sharp detail, natural lighting",
    "oil_painting":    "rendered as an oil painting in the style of classical impressionism",
    "watercolor":      "rendered as a watercolor painting with soft edges and translucent washes",
    "pencil_sketch":   "detailed pencil sketch with expressive cross-hatching and shading",
    "anime":           "anime illustration style, cel-shaded, vibrant colors",
    "dark_fantasy":    "dark fantasy digital art, dramatic lighting, highly detailed",
    "concept_art":     "professional concept art, matte painting style, cinematic composition",
    "ink_wash":        "ink wash painting, East Asian brush art style, minimal and expressive",
}

STYLES = list(STYLE_SUFFIXES.keys())  # exported for model validation


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def generate_images(
    prompt: str,
    n: int = 2,
    aspect_ratio: str = "auto",
    resolution: str = "1k",
    style: Optional[str] = None,
    api_key: Optional[str] = None,
) -> list[str]:
    """
    Generate `n` images from a text prompt using Grok Aurora.

    aspect_ratio  — one of ASPECT_RATIOS (default "auto")
    resolution    — "1k" or "2k" (default "1k")
    style         — one of STYLES; appended to prompt as text (default None)

    Returns a list of image URLs (temporary xAI-hosted URLs).
    Raises RuntimeError on API failure.
    """
    key = api_key or os.getenv("XAI_API_KEY")
    if not key:
        raise RuntimeError("XAI_API_KEY is not set.")

    # Append style suffix to prompt if requested
    final_prompt = prompt
    if style and style in STYLE_SUFFIXES:
        final_prompt = f"{prompt.rstrip('.')} — {STYLE_SUFFIXES[style]}"

    client = OpenAI(api_key=key, base_url=GROK_BASE_URL)

    logger.info(
        f"Generating {n} image(s) | aspect={aspect_ratio} "
        f"res={resolution} style={style or 'none'} | prompt: {final_prompt[:80]}…"
    )

    # aspect_ratio and resolution are xAI-specific extensions to the OpenAI
    # images.generate() spec — must be passed via extra_body.
    response = client.images.generate(
        model=IMAGE_MODEL,
        prompt=final_prompt,
        n=n,
        extra_body={
            "aspect_ratio": aspect_ratio,
            "resolution":   resolution,
        },
    )

    urls = [item.url for item in response.data if item.url]
    logger.info(f"Generated {len(urls)} image(s)")
    return urls


# ---------------------------------------------------------------------------
# Vault-aware prompt builder
# ---------------------------------------------------------------------------

def _style_instruction(style: Optional[str]) -> str:
    """Return a sentence telling the vault prompt builder which style to use."""
    if not style or style not in STYLE_SUFFIXES:
        return ""
    return f"\n- The image should be rendered in this style: {STYLE_SUFFIXES[style]}"


PROMPT_BUILDER_SYSTEM = """You are an expert at writing prompts for AI image generation.
You are helping generate images for a tabletop RPG campaign.

You will be given:
1. A user's description of what they want to visualize.
2. Relevant content from the campaign's Obsidian vault — character sheets,
   location descriptions, faction lore, session notes, etc.
3. An optional style instruction.

Your job is to write ONE rich, detailed image generation prompt that:
- Incorporates specific visual details from the vault content (appearance,
  clothing, colours, setting details, atmosphere, time of day, etc.)
- Captures the tone and genre of the campaign (fantasy, dark, gritty, etc.)
- Follows best practices for the requested art style
- Is 2-4 sentences long — specific and vivid, not vague

Return ONLY the image generation prompt. No preamble, no explanation, no quotes.
"""


def build_vault_prompt(
    description: str,
    vault_context: str,
    style: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Ask Grok chat to craft a detailed image generation prompt by combining
    the user's description with relevant vault content.

    Returns the crafted prompt string (style suffix already embedded).
    """
    key = api_key or os.getenv("XAI_API_KEY")
    if not key:
        raise RuntimeError("XAI_API_KEY is not set.")

    client = OpenAI(api_key=key, base_url=GROK_BASE_URL)

    style_line = _style_instruction(style)

    user_message = f"""## What I want to visualize
{description}

## Relevant vault content
{vault_context}

## Style requirement{style_line if style_line else ": no specific style — use whatever suits the content best"}

Write an image generation prompt based on the above."""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        max_tokens=512,
        messages=[
            {"role": "system", "content": PROMPT_BUILDER_SYSTEM},
            {"role": "user",   "content": user_message},
        ],
    )

    crafted = response.choices[0].message.content.strip()
    logger.info(f"Vault-crafted prompt: {crafted[:100]}…")
    return crafted


# ---------------------------------------------------------------------------
# Image editing  (POST /v1/images/edits)
# ---------------------------------------------------------------------------
# The xAI image edit endpoint uses application/json — not multipart/form-data
# like the OpenAI spec.  This means the OpenAI SDK's client.images.edit()
# method does NOT work here; we call the HTTP API directly with requests.
#
# Pricing: $0.022 per output image ($0.02 output + $0.002 input).
# URLs are temporary — download promptly.
#
# Modes
# -----
#   Single-image edit:
#     image_urls = ["https://..."]     → edits the one image
#
#   Multi-image composite:
#     image_urls = ["https://url1", "https://url2"]
#     → model understands both and follows the prompt
#       e.g. "add the character from the first image into the second scene"
#
#   Multi-turn chaining:
#     Pass the URL returned from a previous edit call as image_urls[0] to
#     refine iteratively — each output becomes the next input.

EDIT_ENDPOINT = f"{GROK_BASE_URL}/images/edits"


def edit_images(
    prompt: str,
    image_urls: list[str],
    n: int = 1,
    aspect_ratio: str = "auto",
    resolution: str = "1k",
    style: Optional[str] = None,
    api_key: Optional[str] = None,
) -> list[str]:
    """
    Edit one or more images using Grok Aurora (grok-imagine-image).

    Sends a POST to /v1/images/edits with application/json.

    Parameters
    ----------
    prompt          Natural-language description of the desired change.
                    Examples:
                      "Render this as an oil painting in the style of impressionism"
                      "Add the character from the first image into the second scene"
                      "Change the lighting to golden-hour sunset"
                      "Keep the face and pose unchanged; swap the armour for dark robes"
    image_urls      List of source image URLs or base64 data URIs.
                    One URL  → single-image edit.
                    Multiple → multi-image compositing.
                    Max 10 images.  Accepts public URLs or base64 data URIs
                    (e.g. "data:image/jpeg;base64,...").
    n               Number of output images to generate (default 1).
    aspect_ratio    Output aspect ratio — same values as generate_images().
    resolution      "1k" or "2k" (default "1k").
    style           Optional style preset — appended to the prompt as text.

    Returns
    -------
    list[str]   Output image URLs (temporary xAI-hosted).

    Raises
    ------
    RuntimeError    on API error or missing credentials.
    ValueError      if image_urls is empty or exceeds 10 items.
    """
    if not image_urls:
        raise ValueError("image_urls must contain at least one URL")
    if len(image_urls) > 10:
        raise ValueError("image_urls must contain 10 or fewer URLs")

    key = api_key or os.getenv("XAI_API_KEY")
    if not key:
        raise RuntimeError("XAI_API_KEY is not set.")

    # Append style suffix to prompt if requested
    final_prompt = prompt
    if style and style in STYLE_SUFFIXES:
        final_prompt = f"{prompt.rstrip('.')} — {STYLE_SUFFIXES[style]}"

    # Build the images array (xAI format)
    images_payload = [{"url": u, "type": "image_url"} for u in image_urls]

    body: dict = {
        "model":        IMAGE_MODEL,
        "prompt":       final_prompt,
        "n":            n,
        "aspect_ratio": aspect_ratio,
        "resolution":   resolution,
    }

    # Single image uses "image" key; multiple uses "images" key
    if len(image_urls) == 1:
        body["image"] = images_payload[0]
    else:
        body["images"] = images_payload

    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {key}",
    }

    logger.info(
        f"Editing {len(image_urls)} image(s) | n={n} aspect={aspect_ratio} "
        f"res={resolution} style={style or 'none'} | prompt: {final_prompt[:80]}…"
    )

    resp = requests.post(EDIT_ENDPOINT, json=body, headers=headers, timeout=120)

    if not resp.ok:
        raise RuntimeError(
            f"xAI image edit error {resp.status_code}: {resp.text[:400]}"
        )

    data = resp.json()
    urls = [item.get("url", "") for item in data.get("data", []) if item.get("url")]

    logger.info(f"Image edit done | {len(urls)} image(s) returned")
    return urls
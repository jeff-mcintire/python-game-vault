"""
image_gen.py — xAI Grok image generation.

Two modes:

  generate(prompt, n)
    Direct generation — takes a prompt string, returns n image URLs.
    Uses POST https://api.x.ai/v1/images/generations via the OpenAI SDK.

  build_vault_prompt(description, vault_context, api_key)
    Vault-aware prompt builder — feeds the user's description plus
    relevant vault file content to Grok chat, which crafts a rich,
    detailed image generation prompt grounded in the campaign lore.
    Returns the crafted prompt string.

Model: grok-2-image-1212  ($0.07/image)
"""

import logging
import os
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

IMAGE_MODEL = "grok-imagine-image"
CHAT_MODEL  = "grok-4-1-fast-reasoning"
GROK_BASE_URL = "https://api.x.ai/v1"


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def generate_images(
    prompt: str,
    n: int = 2,
    api_key: Optional[str] = None,
) -> list[str]:
    """
    Generate `n` images from a text prompt using Grok Aurora.

    Returns a list of image URLs (temporary xAI-hosted URLs).
    Raises RuntimeError on API failure.
    """
    key = api_key or os.getenv("XAI_API_KEY")
    if not key:
        raise RuntimeError("XAI_API_KEY is not set.")

    client = OpenAI(api_key=key, base_url=GROK_BASE_URL)

    logger.info(f"Generating {n} image(s) with prompt: {prompt[:80]}…")

    response = client.images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
        n=n,
    )

    urls = [item.url for item in response.data if item.url]
    logger.info(f"Generated {len(urls)} image(s)")
    return urls


# ---------------------------------------------------------------------------
# Vault-aware prompt builder
# ---------------------------------------------------------------------------

PROMPT_BUILDER_SYSTEM = """You are an expert at writing prompts for AI image generation.
You are helping generate images for a tabletop RPG campaign.

You will be given:
1. A user's description of what they want to visualize.
2. Relevant content from the campaign's Obsidian vault — character sheets,
   location descriptions, faction lore, session notes, etc.

Your job is to write ONE rich, detailed image generation prompt that:
- Incorporates specific visual details from the vault content (appearance,
  clothing, colours, setting details, atmosphere, time of day, etc.)
- Captures the tone and genre of the campaign (fantasy, dark, gritty, etc.)
- Follows best practices for photorealistic or painterly fantasy image prompts:
  include subject, setting, lighting, mood, art style, and camera/composition cues
- Is 2-4 sentences long — specific and vivid, not vague

Return ONLY the image generation prompt. No preamble, no explanation, no quotes.
"""


def build_vault_prompt(
    description: str,
    vault_context: str,
    api_key: Optional[str] = None,
) -> str:
    """
    Ask Grok chat to craft a detailed image generation prompt by combining
    the user's description with relevant vault content.

    Returns the crafted prompt string.
    """
    key = api_key or os.getenv("XAI_API_KEY")
    if not key:
        raise RuntimeError("XAI_API_KEY is not set.")

    client = OpenAI(api_key=key, base_url=GROK_BASE_URL)

    user_message = f"""## What I want to visualize
{description}

## Relevant vault content
{vault_context}

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

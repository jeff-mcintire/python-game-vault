"""
vision.py — Image analysis using Claude or Grok vision models.

Sends an image URL to the chosen LLM and returns two generation-ready
prompt descriptions:
  1. Aurora prompt  — phrased for Grok Imagine / DALL-E style generators
  2. Flux/SD3 prompt — dense, tag-style prompt optimised for Flux 2 Pro / SD3

Both providers accept image URLs directly — no local download needed.

Claude vision  → claude-opus-4-5 (or any claude-3+ model)
               Content block: {"type": "image", "source": {"type": "url", "url": "..."}}

Grok vision    → grok-2-vision (xAI vision model)
               OpenAI-compat:  content array with {"type": "image_url", ...}
"""

import logging
import os
from typing import Optional

import anthropic
from openai import OpenAI

logger = logging.getLogger(__name__)

GROK_BASE_URL = "https://api.x.ai/v1"
GROK_VISION_MODEL  = "grok-4"
CLAUDE_VISION_MODEL = "claude-opus-4-5"

# ── The two prompts ──────────────────────────────────────────────────────────

AURORA_PROMPT = (
    "Describe the main person in the attached image in a way that Grok Imagine "
    "could reproduce them. Include physical appearance (face shape, eye color, "
    "hair color, length and style, skin tone, build), clothing and accessories, "
    "pose, expression, and the overall mood or atmosphere of the image. "
    "Write it as a natural image generation prompt — vivid, concrete, and specific. "
    "Do not include any names or identifying information."
)

FLUX_PROMPT = (
    "Now give me a second version of that description, this time optimised for "
    "Flux 2 Pro / Stable Diffusion 3. Use the dense tag-and-phrase style these "
    "models respond best to: lead with the most important subject descriptors, "
    "then physical details, then clothing, then lighting/style/quality boosters. "
    "Separate concepts with commas. Include quality tags like "
    "'highly detailed, sharp focus, 8k, photorealistic'. "
    "No full sentences — pure optimised prompt tokens."
)

COMBINED_SYSTEM = (
    "You are an expert at analysing images and writing precise, detailed prompts "
    "for AI image generators. You describe people objectively and specifically, "
    "focusing on visual attributes that image models can reproduce. "
    "You never include real names or personally identifying information."
)


def _parse_sections(text: str) -> tuple[str, str]:
    """
    Parse the LLM response into (aurora_prompt, flux_prompt).
    The model is asked to label sections; we split on the header.
    Falls back gracefully if the model doesn't follow the format exactly.
    """
    aurora = ""
    flux = ""

    # Try splitting on our section markers
    markers = [
        ("AURORA PROMPT", "FLUX PROMPT"),
        ("GROK IMAGINE PROMPT", "FLUX"),
        ("1.", "2."),
        ("**1", "**2"),
    ]
    for m1, m2 in markers:
        upper = text.upper()
        i1 = upper.find(m1.upper())
        i2 = upper.find(m2.upper())
        if i1 != -1 and i2 != -1 and i2 > i1:
            aurora = text[i1:i2].strip()
            flux   = text[i2:].strip()
            # Strip the header lines themselves
            aurora = "\n".join(aurora.splitlines()[1:]).strip()
            flux   = "\n".join(flux.splitlines()[1:]).strip()
            break

    if not aurora and not flux:
        # No recognised markers — split roughly in half
        lines = text.strip().splitlines()
        mid = len(lines) // 2
        aurora = "\n".join(lines[:mid]).strip()
        flux   = "\n".join(lines[mid:]).strip()

    return aurora or text.strip(), flux or text.strip()


def analyze_image_claude(
    image_url: str,
    api_key: Optional[str] = None,
    model: str = CLAUDE_VISION_MODEL,
) -> tuple[str, str]:
    """
    Send image_url to Claude with both prompts in a single multi-turn exchange.

    Returns (aurora_prompt_text, flux_prompt_text).
    """
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    client = anthropic.Anthropic(api_key=key)

    logger.info(f"Claude vision | model={model} url={image_url[:60]}…")

    # Turn 1 — Aurora description
    resp1 = client.messages.create(
        model=model,
        max_tokens=1024,
        system=COMBINED_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "url", "url": image_url},
                    },
                    {
                        "type": "text",
                        "text": AURORA_PROMPT,
                    },
                ],
            }
        ],
    )
    aurora_text = resp1.content[0].text.strip()
    logger.info("Claude vision — Aurora description done")

    # Turn 2 — Flux/SD3 re-optimisation (no need to re-send image)
    resp2 = client.messages.create(
        model=model,
        max_tokens=1024,
        system=COMBINED_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "url", "url": image_url},
                    },
                    {"type": "text", "text": AURORA_PROMPT},
                ],
            },
            {"role": "assistant", "content": aurora_text},
            {"role": "user",      "content": FLUX_PROMPT},
        ],
    )
    flux_text = resp2.content[0].text.strip()
    logger.info("Claude vision — Flux description done")

    return aurora_text, flux_text


def analyze_image_grok(
    image_url: str,
    api_key: Optional[str] = None,
    model: str = GROK_VISION_MODEL,
) -> tuple[str, str]:
    """
    Send image_url to Grok vision (grok-2-vision) with both prompts.

    Returns (aurora_prompt_text, flux_prompt_text).
    """
    key = api_key or os.getenv("XAI_API_KEY")
    if not key:
        raise RuntimeError("XAI_API_KEY is not set.")

    client = OpenAI(api_key=key, base_url=GROK_BASE_URL)

    logger.info(f"Grok vision | model={model} url={image_url[:60]}…")

    # Turn 1 — Aurora description
    messages: list[dict] = [
        {"role": "system", "content": COMBINED_SYSTEM},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_url, "detail": "high"},
                },
                {"type": "text", "text": AURORA_PROMPT},
            ],
        },
    ]

    resp1 = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=messages,
    )
    aurora_text = resp1.choices[0].message.content.strip()
    logger.info("Grok vision — Aurora description done")

    # Turn 2 — continue conversation for Flux prompt
    messages.append({"role": "assistant", "content": aurora_text})
    messages.append({"role": "user",      "content": FLUX_PROMPT})

    resp2 = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=messages,
    )
    flux_text = resp2.choices[0].message.content.strip()
    logger.info("Grok vision — Flux description done")

    return aurora_text, flux_text
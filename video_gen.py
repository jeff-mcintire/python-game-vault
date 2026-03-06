"""
video_gen.py — xAI Grok video generation.

Three generation modes:

  generate_video(prompt, duration, aspect_ratio, resolution, image_url)
    Text-to-video — prompt → video.
    Optionally supply image_url to use image-to-video mode instead.
    Blocks (polls) until the video is ready or the timeout is reached.

  edit_video(video_url, prompt)
    Video editing — supply an existing video URL and a change description.
    The model restyle/edits the video per the prompt.
    Input video must be ≤ 8.7 seconds.  Duration is not user-configurable
    in edit mode; the output matches the input's length.
    Blocks (polls) until done.

  start_video(...) / poll_video(request_id)
    Non-blocking variant: start_video returns a request_id immediately.
    Use poll_video to check / wait for the result.

  build_vault_video_prompt(description, vault_context, style)
    Vault-aware prompt builder — feeds the user's description plus relevant
    vault file content to Grok chat, which crafts a rich, cinematic prompt
    for video generation grounded in the campaign lore.
    Returns the crafted prompt string.

API details (grok-imagine-video):
  Endpoint: POST https://api.x.ai/v1/videos/generations
  Status:   GET  https://api.x.ai/v1/videos/{request_id}
  Params:
    prompt        (required)
    model         (default: grok-imagine-video)
    duration      1–15 s (text/image-to-video only; ignored for editing)
    aspect_ratio  "16:9" default; see VIDEO_ASPECT_RATIOS
    resolution    "480p" | "720p" | "1080p"
    image.url     source image URL for image-to-video mode
    video_url     source video URL for edit mode (max 8.7 s)

  Generation is async: POST → request_id, then poll until status == "done".
"""

import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

VIDEO_MODEL   = "grok-imagine-video"
CHAT_MODEL    = "grok-4-1-fast-reasoning"
GROK_BASE_URL = "https://api.x.ai/v1"

# ---------------------------------------------------------------------------
# Valid values — exported for models.py validation
# ---------------------------------------------------------------------------

VIDEO_ASPECT_RATIOS = [
    "16:9",   # default — landscape widescreen
    "9:16",   # vertical / portrait
    "4:3",    # classic / presentation
    "3:4",    # portrait card
    "1:1",    # square
]

VIDEO_RESOLUTIONS = ["480p", "720p", "1080p"]

# How long (s) to wait for a video before giving up, and polling interval
DEFAULT_TIMEOUT_SECONDS  = 300   # 5 minutes
DEFAULT_POLL_INTERVAL    = 5     # check every 5 seconds


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _headers(api_key: str) -> dict:
    return {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {api_key}",
    }


def _resolve_key(api_key: Optional[str]) -> str:
    key = api_key or os.getenv("XAI_API_KEY")
    if not key:
        raise RuntimeError("XAI_API_KEY is not set.")
    return key


# ---------------------------------------------------------------------------
# Non-blocking start
# ---------------------------------------------------------------------------

def start_video(
    prompt: str,
    duration: int = 8,
    aspect_ratio: str = "16:9",
    resolution: str = "720p",
    image_url: Optional[str] = None,
    video_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Submit a video generation or editing request and return the request_id
    immediately (non-blocking).

    Pass image_url  to animate a still image (image-to-video).
    Pass video_url  to edit an existing video (max 8.7 s input).

    Returns: request_id string.
    """
    key = _resolve_key(api_key)

    body: dict = {
        "model":        VIDEO_MODEL,
        "prompt":       prompt,
        "aspect_ratio": aspect_ratio,
        "resolution":   resolution,
    }

    if video_url:
        # Edit mode — duration is ignored by the API; output matches input length
        body["video_url"] = video_url
    else:
        body["duration"] = max(1, min(15, duration))
        if image_url:
            body["image"] = {"url": image_url}

    logger.info(
        f"Starting video gen | mode={'edit' if video_url else ('img2vid' if image_url else 'txt2vid')} "
        f"dur={duration}s aspect={aspect_ratio} res={resolution} | prompt: {prompt[:80]}…"
    )

    resp = requests.post(
        f"{GROK_BASE_URL}/videos/generations",
        headers=_headers(key),
        json=body,
        timeout=30,
    )

    if not resp.ok:
        raise RuntimeError(
            f"xAI video API error {resp.status_code}: {resp.text[:400]}"
        )

    data = resp.json()
    request_id = data.get("request_id")
    if not request_id:
        raise RuntimeError(f"No request_id in response: {data}")

    logger.info(f"Video generation queued — request_id: {request_id}")
    return request_id


# ---------------------------------------------------------------------------
# Status poll
# ---------------------------------------------------------------------------

def poll_video(
    request_id: str,
    api_key: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    interval: int = DEFAULT_POLL_INTERVAL,
) -> dict:
    """
    Poll GET /v1/videos/{request_id} until status == "done" or "error",
    or until `timeout` seconds elapse.

    Returns the final response dict:
    {
        "status": "done",
        "video": {
            "url": "https://vidgen.x.ai/...",
            "duration": 8,
            "respect_moderation": true
        },
        "model": "grok-imagine-video"
    }

    Raises RuntimeError on error status or timeout.
    """
    key = _resolve_key(api_key)
    deadline = time.time() + timeout
    poll_url  = f"{GROK_BASE_URL}/videos/{request_id}"
    auth      = {"Authorization": f"Bearer {key}"}

    logger.info(f"Polling video {request_id} (timeout {timeout}s, every {interval}s)…")

    while time.time() < deadline:
        resp = requests.get(poll_url, headers=auth, timeout=15)

        if not resp.ok:
            raise RuntimeError(
                f"xAI video status error {resp.status_code}: {resp.text[:400]}"
            )

        data   = resp.json()
        status = data.get("status", "unknown")

        if status == "done":
            logger.info(f"Video ready — {data.get('video', {}).get('url', '')[:60]}…")
            return data

        if status == "error":
            raise RuntimeError(
                f"Video generation failed (request_id={request_id}): {data}"
            )

        # Still pending — wait before next poll
        logger.debug(f"Video status: {status} — retrying in {interval}s…")
        time.sleep(interval)

    raise TimeoutError(
        f"Video generation timed out after {timeout}s (request_id={request_id})"
    )


# ---------------------------------------------------------------------------
# Blocking convenience wrappers
# ---------------------------------------------------------------------------

def generate_video(
    prompt: str,
    duration: int = 8,
    aspect_ratio: str = "16:9",
    resolution: str = "720p",
    image_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """
    Generate a video (text-to-video or image-to-video) and wait for the result.

    prompt       — what the video should show
    duration     — length in seconds (1–15, ignored in edit mode)
    aspect_ratio — one of VIDEO_ASPECT_RATIOS
    resolution   — "480p" | "720p" | "1080p"
    image_url    — optional; if set, animates this image (image-to-video mode)

    Returns a dict with keys:
        video_url     str
        duration      int
        request_id    str
        moderated     bool
    """
    request_id = start_video(
        prompt=prompt,
        duration=duration,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        image_url=image_url,
        api_key=api_key,
    )

    result = poll_video(request_id, api_key=api_key, timeout=timeout)

    video  = result.get("video", {})
    return {
        "video_url":  video.get("url", ""),
        "duration":   video.get("duration", duration),
        "request_id": request_id,
        "moderated":  video.get("respect_moderation", True),
    }


def edit_video(
    video_url: str,
    prompt: str,
    api_key: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """
    Edit an existing video by describing the desired changes.

    video_url — publicly accessible URL to the source video (max 8.7 s)
    prompt    — description of the desired changes
                ("make the sky stormy", "remove the background figures", etc.)

    Duration is set by the source video; you cannot specify it.

    Returns a dict with keys:
        video_url     str
        duration      int
        request_id    str
        moderated     bool
    """
    request_id = start_video(
        prompt=prompt,
        video_url=video_url,
        api_key=api_key,
    )

    result = poll_video(request_id, api_key=api_key, timeout=timeout)

    video  = result.get("video", {})
    return {
        "video_url":  video.get("url", ""),
        "duration":   video.get("duration", 0),
        "request_id": request_id,
        "moderated":  video.get("respect_moderation", True),
    }


# ---------------------------------------------------------------------------
# Vault-aware prompt builder
# ---------------------------------------------------------------------------

VAULT_VIDEO_PROMPT_SYSTEM = """\
You are an expert at writing prompts for AI video generation.
You are helping generate short cinematic video clips for a tabletop RPG campaign.

You will be given:
1. A user's description of what they want to visualize as a video.
2. Relevant content from the campaign's Obsidian vault — character sheets,
   location descriptions, faction lore, session notes, etc.
3. Optional style direction.

Your job is to write ONE rich, detailed video generation prompt that:
- Incorporates specific visual details from the vault content (appearance,
  clothing, colours, setting details, atmosphere, camera movement, lighting)
- Describes motion and action — this is a VIDEO, not a still image
- Suggests a cinematic camera movement where appropriate (slow push in, orbital,
  handheld, aerial, etc.)
- Captures the tone and genre of the campaign (fantasy, dark, gritty, epic, etc.)
- Is 2-4 sentences long — specific and vivid, not vague

Return ONLY the video generation prompt. No preamble, no explanation, no quotes.
"""


def build_vault_video_prompt(
    description: str,
    vault_context: str,
    style: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Ask Grok chat to craft a detailed video generation prompt by combining
    the user's description with relevant vault content.

    Returns the crafted prompt string.
    """
    from openai import OpenAI  # local import — same pattern as image_gen.py

    key = _resolve_key(api_key)
    client = OpenAI(api_key=key, base_url=GROK_BASE_URL)

    style_line = f"\n- Visual style direction: {style}" if style else ""

    user_message = f"""## What I want to visualize as a video
{description}

## Relevant vault content
{vault_context}

## Style requirement{style_line if style_line else ": no specific style — use whatever suits the campaign tone best"}

Write a cinematic video generation prompt based on the above.
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        max_tokens=512,
        messages=[
            {"role": "system", "content": VAULT_VIDEO_PROMPT_SYSTEM},
            {"role": "user",   "content": user_message},
        ],
    )

    crafted = response.choices[0].message.content.strip()
    logger.info(f"Vault-crafted video prompt: {crafted[:100]}…")
    return crafted
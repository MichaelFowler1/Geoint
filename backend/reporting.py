from __future__ import annotations

import base64
import io
import logging

from .config import get_settings

logger = logging.getLogger("geoint.reporting")

ANALYST_SYSTEM = (
    "You are an imagery intelligence analyst. Write concise, factual, "
    "non-speculative assessments in the style of a SITREP. Do not invent details "
    "that are not supported by the inputs."
)


def _client():
    s = get_settings()
    if not s.openai_api_key:
        return None
    try:
        from openai import OpenAI  # noqa: PLC0415
    except ImportError:
        logger.warning("openai package not installed")
        return None
    return OpenAI(api_key=s.openai_api_key)


def _encode_png(image_path: str) -> str:
    """Load any raster (incl. GeoTIFF) and return it base64-encoded as PNG.

    OpenAI's vision endpoint only accepts png/jpeg/gif/webp, so a GeoTIFF must be
    converted first. Pillow reads the pixels and ignores the geo tags.
    """
    from PIL import Image  # noqa: PLC0415

    img = Image.open(image_path).convert("RGB")
    img.thumbnail((2048, 2048))  # cap payload / token cost
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def scene_assessment(image_path: str) -> str:
    """Qualitative scene read via GPT vision ('the rest', where GPT is strong)."""
    client = _client()
    if client is None:
        return "[scene assessment unavailable - set OPENAI_API_KEY]"
    s = get_settings()
    b64 = _encode_png(image_path)
    try:
        resp = client.chat.completions.create(
            model=s.openai_vision_model,
            messages=[
                {"role": "system", "content": ANALYST_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Provide a brief scene-level assessment of this "
                                "overhead image: facility type, layout, and any "
                                "notable features. 2-4 sentences."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                },
            ],
            max_tokens=300,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("scene_assessment failed: %s", exc)
        return f"[scene assessment error: {exc}]"


def generate_sitrep(
    counts: dict[str, int], scene: str, location: str | None = None
) -> str:
    """Fuse precise detector counts with the qualitative scene read into a SITREP."""
    client = _client()
    count_line = (
        ", ".join(f"{n}x {label}" for label, n in counts.items())
        or "no objects detected"
    )
    if client is None:
        loc = f" at {location}" if location else ""
        return f"SITREP{loc}: detector reports {count_line}. Scene: {scene}"
    s = get_settings()
    prompt = (
        f"Detector counts: {count_line}.\n"
        f"Scene assessment: {scene}.\n"
        f"Location: {location or 'unspecified'}.\n"
        "Write a 2-3 sentence SITREP integrating the precise counts with the "
        "scene context."
    )
    try:
        resp = client.chat.completions.create(
            model=s.openai_report_model,
            messages=[
                {"role": "system", "content": ANALYST_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=250,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("generate_sitrep failed: %s", exc)
        return f"SITREP: detector reports {count_line}. Scene: {scene}"
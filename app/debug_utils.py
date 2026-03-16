import json
import os
from pathlib import Path
from typing import Any

from PIL import Image


DEBUG_ENABLED = os.getenv("STORYAI_DEBUG", "true").lower() == "true"


def debug_print(title: str, payload: Any | None = None) -> None:
    """Print a structured debug block when terminal debugging is enabled."""

    if not DEBUG_ENABLED:
        return

    divider = "=" * 24
    print(f"\n{divider} {title} {divider}", flush=True)
    if payload is None:
        return
    if isinstance(payload, str):
        print(payload, flush=True)
        return
    print(_to_pretty_json(payload), flush=True)


def _to_pretty_json(payload: Any) -> str:
    """Serialize debug data in a copy-paste friendly way."""

    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def summarize_path(path: str | Path) -> dict[str, Any]:
    """Return basic metadata for a file path."""

    resolved = Path(path)
    info: dict[str, Any] = {
        "path": str(resolved.resolve()) if resolved.exists() else str(resolved),
        "exists": resolved.exists(),
    }
    if resolved.exists():
        stat = resolved.stat()
        info["size_bytes"] = stat.st_size
    return info


def summarize_image(path: str | Path) -> dict[str, Any]:
    """Return file and dimension metadata for an image."""

    resolved = Path(path)
    summary = summarize_path(resolved)
    if resolved.exists():
        with Image.open(resolved) as image:
            summary.update(
                {
                    "format": image.format,
                    "mode": image.mode,
                    "width": image.width,
                    "height": image.height,
                }
            )
    return summary


def mask_api_key(api_key: str | None) -> str:
    """Return a safe representation of the API key for logs."""

    if not api_key:
        return "<missing>"
    if len(api_key) <= 12:
        return "<redacted>"
    return f"{api_key[:7]}...{api_key[-4:]}"

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps

from .debug_utils import debug_print, mask_api_key, summarize_image, summarize_path
from .openai_api import build_client, generate_page_image, generate_story_package
from .schemas import StoryPackage


DEFAULT_RUNS_DIR = Path(os.getenv("STORYAI_RUNS_DIR", "runs"))
MAX_IMAGE_SIZE = 1536


def _new_run_dir(base_dir: Path = DEFAULT_RUNS_DIR) -> Path:
    """Create a timestamped artifact directory for one app run."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{timestamp}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def normalize_image(source_path: Path, output_path: Path) -> None:
    """Normalize uploads into an RGB working PNG with a bounded size."""

    with Image.open(source_path) as image:
        working = ImageOps.exif_transpose(image).convert("RGB")
        working.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
        working.save(output_path, format="PNG")


def run_story_package_pipeline(image_path: str, api_key: str | None = None) -> dict:
    """Run the phase-one pipeline and save all generated artifacts."""

    source_path = Path(image_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Uploaded image not found: {source_path}")

    run_dir = _new_run_dir()
    copied_input_path = run_dir / f"input_image{source_path.suffix.lower() or '.png'}"
    working_image_path = run_dir / "working_image.png"

    debug_print(
        "PIPELINE START",
        {
            "source_image": summarize_image(source_path),
            "run_dir": str(run_dir.resolve()),
            "story_model": os.getenv("STORYAI_STORY_MODEL", "gpt-4o-mini"),
            "api_key": mask_api_key(api_key or os.getenv("OPENAI_API_KEY")),
        },
    )

    shutil.copy2(source_path, copied_input_path)
    normalize_image(copied_input_path, working_image_path)
    debug_print("COPIED INPUT IMAGE", summarize_image(copied_input_path))
    debug_print("NORMALIZED WORKING IMAGE", summarize_image(working_image_path))

    client = build_client(api_key=api_key)
    story_package, raw_response = generate_story_package(client, working_image_path)
    page_images = _generate_page_images(client, story_package, run_dir)

    story_package_path = run_dir / "story_package.json"
    openai_response_path = run_dir / "openai_response.json"
    page_image_manifest_path = run_dir / "page_images.json"
    _write_json(story_package_path, story_package.model_dump(mode="json"))
    _write_json(openai_response_path, raw_response)
    _write_json(page_image_manifest_path, {"pages": page_images})

    debug_print(
        "ARTIFACTS WRITTEN",
        {
            "story_package_json": summarize_path(story_package_path),
            "openai_response_json": summarize_path(openai_response_path),
            "page_image_manifest_json": summarize_path(page_image_manifest_path),
        },
    )

    return {
        "run_dir": str(run_dir.resolve()),
        "input_image": str(copied_input_path.resolve()),
        "working_image": str(working_image_path.resolve()),
        "openai_response_path": str(openai_response_path.resolve()),
        "story_package_path": str(story_package_path.resolve()),
        "page_image_manifest_path": str(page_image_manifest_path.resolve()),
        "page_images": page_images,
        "story_package": story_package,
    }


def _write_json(output_path: Path, payload: StoryPackage | dict) -> None:
    """Write indented JSON using UTF-8 encoding."""

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _generate_page_images(client, story_package: StoryPackage, run_dir: Path) -> list[dict]:
    """Generate one page image per story part and save prompt artifacts."""

    pages = [
        ("page_1", story_package.part_1),
        ("page_2", story_package.part_2),
        ("page_3", story_package.part_3),
    ]
    generated_pages: list[dict] = []

    for page_name, story_part in pages:
        output_path = run_dir / f"{page_name}.png"
        prompt_path = run_dir / f"{page_name}_prompt.txt"
        final_prompt, response_summary = generate_page_image(
            client=client,
            visual_canon=story_package.visual_canon,
            page_prompt=story_part.image_prompt,
            output_path=output_path,
        )
        prompt_path.write_text(final_prompt, encoding="utf-8")

        generated_page = {
            "page": page_name,
            "story_text": story_part.text,
            "story_prompt": story_part.image_prompt,
            "final_prompt": final_prompt,
            "image_path": str(output_path.resolve()),
            "prompt_path": str(prompt_path.resolve()),
            "response": response_summary,
        }
        generated_pages.append(generated_page)

    debug_print("PAGE IMAGES", generated_pages)
    return generated_pages

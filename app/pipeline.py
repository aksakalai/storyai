import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps

from .debug_utils import (
    debug_print,
    mask_api_key,
    summarize_image,
    summarize_path,
)
from .openai_api import (
    build_client,
    generate_page_image,
    generate_story_package,
    synthesize_narration,
    transcribe_audio_with_word_timestamps,
)
from .schemas import StoryPackage
from .video import concatenate_page_videos, render_page_video


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
    """Run the full story-to-video pipeline and save all generated artifacts."""

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
    page_audio = _generate_page_audio(client, story_package, run_dir)
    page_timestamps = _generate_page_timestamps(client, page_audio, run_dir)
    page_videos, final_video_summary = _render_page_videos(
        page_images=page_images,
        page_audio=page_audio,
        page_timestamps=page_timestamps,
        run_dir=run_dir,
    )

    story_package_path = run_dir / "story_package.json"
    openai_response_path = run_dir / "openai_response.json"
    page_image_manifest_path = run_dir / "page_images.json"
    page_audio_manifest_path = run_dir / "page_audio.json"
    page_timestamps_manifest_path = run_dir / "page_timestamps.json"
    page_video_manifest_path = run_dir / "page_videos.json"
    _write_json(story_package_path, story_package.model_dump(mode="json"))
    _write_json(openai_response_path, raw_response)
    _write_json(page_image_manifest_path, {"pages": page_images})
    _write_json(page_audio_manifest_path, {"pages": page_audio})
    _write_json(page_timestamps_manifest_path, {"pages": page_timestamps})
    _write_json(
        page_video_manifest_path,
        {
            "pages": page_videos,
            "final_video": final_video_summary,
        },
    )

    debug_print(
        "ARTIFACTS WRITTEN",
        {
            "story_package_json": summarize_path(story_package_path),
            "openai_response_json": summarize_path(openai_response_path),
            "page_image_manifest_json": summarize_path(page_image_manifest_path),
            "page_audio_manifest_json": summarize_path(page_audio_manifest_path),
            "page_timestamps_manifest_json": summarize_path(page_timestamps_manifest_path),
            "page_video_manifest_json": summarize_path(page_video_manifest_path),
            "final_video_mp4": summarize_path(final_video_summary["video_path"]),
        },
    )

    return {
        "run_dir": str(run_dir.resolve()),
        "input_image": str(copied_input_path.resolve()),
        "working_image": str(working_image_path.resolve()),
        "openai_response_path": str(openai_response_path.resolve()),
        "story_package_path": str(story_package_path.resolve()),
        "page_image_manifest_path": str(page_image_manifest_path.resolve()),
        "page_audio_manifest_path": str(page_audio_manifest_path.resolve()),
        "page_timestamps_manifest_path": str(page_timestamps_manifest_path.resolve()),
        "page_video_manifest_path": str(page_video_manifest_path.resolve()),
        "final_video_path": final_video_summary["video_path"],
        "page_images": page_images,
        "page_audio": page_audio,
        "page_timestamps": page_timestamps,
        "page_videos": page_videos,
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


def _generate_page_audio(client, story_package: StoryPackage, run_dir: Path) -> list[dict]:
    """Generate one narration clip per story part and save audio artifacts."""

    pages = [
        ("page_1", story_package.part_1),
        ("page_2", story_package.part_2),
        ("page_3", story_package.part_3),
    ]
    generated_audio: list[dict] = []

    for page_name, story_part in pages:
        output_path = run_dir / f"{page_name}_audio.wav"
        response_summary = synthesize_narration(
            client=client,
            text=story_part.text,
            output_path=output_path,
        )
        generated_audio.append(
            {
                "page": page_name,
                "story_text": story_part.text,
                "audio_path": str(output_path.resolve()),
                "response": response_summary,
            }
        )

    debug_print("PAGE AUDIO", generated_audio)
    return generated_audio


def _generate_page_timestamps(client, page_audio: list[dict], run_dir: Path) -> list[dict]:
    """Transcribe each narration clip and save page-level timing artifacts."""

    generated_timestamps: list[dict] = []

    for page_audio_item in page_audio:
        page_name = page_audio_item["page"]
        audio_path = Path(page_audio_item["audio_path"])
        output_path = run_dir / f"{page_name}_timestamps.json"
        raw_response = transcribe_audio_with_word_timestamps(
            client=client,
            audio_path=audio_path,
            expected_text=page_audio_item["story_text"],
        )
        _write_json(output_path, raw_response)

        generated_timestamps.append(
            {
                "page": page_name,
                "audio_path": str(audio_path.resolve()),
                "timestamps_path": str(output_path.resolve()),
                "story_text": page_audio_item["story_text"],
                "text": raw_response.get("text", ""),
                "words": raw_response.get("words", []) or [],
                "segments": raw_response.get("segments", []) or [],
                "duration_seconds": raw_response.get("duration"),
            }
        )

    debug_print("PAGE TIMESTAMPS", generated_timestamps)
    return generated_timestamps


def _render_page_videos(
    page_images: list[dict],
    page_audio: list[dict],
    page_timestamps: list[dict],
    run_dir: Path,
) -> tuple[list[dict], dict]:
    """Render one page clip per story part, then concatenate the final video."""

    image_by_page = {page["page"]: page for page in page_images}
    audio_by_page = {page["page"]: page for page in page_audio}
    timestamps_by_page = {page["page"]: page for page in page_timestamps}
    page_videos: list[dict] = []
    output_paths: list[Path] = []

    for page_name in ("page_1", "page_2", "page_3"):
        image_item = image_by_page[page_name]
        audio_item = audio_by_page[page_name]
        timestamp_item = timestamps_by_page[page_name]
        output_path = run_dir / f"{page_name}.mp4"
        render_summary = render_page_video(
            image_path=Path(image_item["image_path"]),
            audio_path=Path(audio_item["audio_path"]),
            story_text=audio_item["story_text"],
            words=timestamp_item["words"],
            duration_seconds=float(timestamp_item["duration_seconds"] or 0.0),
            output_path=output_path,
        )
        page_videos.append(
            {
                "page": page_name,
                "story_text": audio_item["story_text"],
                "image_path": image_item["image_path"],
                "audio_path": audio_item["audio_path"],
                "timestamps_path": timestamp_item["timestamps_path"],
                "subtitle_script_path": render_summary["subtitle_script_path"],
                "video_path": render_summary["video_path"],
                "response": render_summary["response"],
            }
        )
        output_paths.append(output_path)

    debug_print("PAGE VIDEOS", page_videos)

    final_output_path = run_dir / "final_story.mp4"
    final_video_response = concatenate_page_videos(output_paths, final_output_path)
    final_video_summary = {
        "video_path": str(final_output_path.resolve()),
        "response": final_video_response,
    }
    debug_print("FINAL VIDEO", final_video_summary)
    return page_videos, final_video_summary

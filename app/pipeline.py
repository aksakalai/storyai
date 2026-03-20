import json
import os
import shutil
import wave
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps

from .debug_utils import (
    debug_print,
    mask_api_key,
    summarize_image,
    summarize_path,
    summarize_wav_audio,
)
from .openai_api import (
    build_client,
    generate_page_image,
    generate_story_package,
    synthesize_narration,
    transcribe_audio_with_word_timestamps,
)
from .schemas import StoryPackage
from .text_utils import sanitize_narration_text
from .video import concatenate_page_videos, render_page_video


DEFAULT_RUNS_DIR = Path(os.getenv("STORYAI_RUNS_DIR", "runs"))
MAX_IMAGE_SIZE = 1536
TITLE_PAUSE_SECONDS = 2.0
ProgressCallback = Callable[[float, str], None]


def _clear_previous_runs(base_dir: Path = DEFAULT_RUNS_DIR) -> None:
    """Remove previous run directories so the latest upload owns the output area."""

    if not base_dir.exists():
        return

    for child in base_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)


def _new_run_dir(base_dir: Path = DEFAULT_RUNS_DIR) -> Path:
    """Create a timestamped artifact directory for one app run."""

    base_dir.mkdir(parents=True, exist_ok=True)
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


def _sanitize_for_narration(text: str) -> str:
    """Normalize visible story text into narration-safe text."""

    return sanitize_narration_text(text) or str(text or "").strip()


def _build_page_one_narration(title_text: str, story_text: str) -> tuple[str, str, str]:
    """Compose the first-page narration so the title is spoken before the story."""

    clean_title = _sanitize_for_narration(title_text)
    clean_story = _sanitize_for_narration(story_text)
    if clean_title and clean_story:
        separator = "" if clean_title.endswith((".", "!", "?")) else "."
        full_narration = f"{clean_title}{separator} {clean_story}".strip()
    else:
        full_narration = clean_title or clean_story
    return clean_title, clean_story, full_narration


def _combine_wav_with_silence(
    first_audio_path: Path,
    second_audio_path: Path,
    output_path: Path,
    silence_seconds: float = TITLE_PAUSE_SECONDS,
) -> dict:
    """Concatenate two WAV files with a deterministic silent gap between them."""

    with wave.open(str(first_audio_path), "rb") as first_audio, wave.open(
        str(second_audio_path),
        "rb",
    ) as second_audio:
        first_signature = (
            first_audio.getnchannels(),
            first_audio.getsampwidth(),
            first_audio.getframerate(),
            first_audio.getcomptype(),
        )
        second_signature = (
            second_audio.getnchannels(),
            second_audio.getsampwidth(),
            second_audio.getframerate(),
            second_audio.getcomptype(),
        )
        if first_signature != second_signature:
            raise RuntimeError("Title and story narration WAV settings do not match.")

        channels = first_audio.getnchannels()
        sample_width = first_audio.getsampwidth()
        frame_rate = first_audio.getframerate()
        silence_frame_count = max(int(round(frame_rate * silence_seconds)), 0)
        silence_bytes = b"\x00" * silence_frame_count * channels * sample_width

        with wave.open(str(output_path), "wb") as combined_audio:
            combined_audio.setnchannels(channels)
            combined_audio.setsampwidth(sample_width)
            combined_audio.setframerate(frame_rate)
            combined_audio.writeframes(first_audio.readframes(first_audio.getnframes()))
            combined_audio.writeframes(silence_bytes)
            combined_audio.writeframes(second_audio.readframes(second_audio.getnframes()))

    summary = summarize_wav_audio(output_path)
    summary["pause_seconds"] = silence_seconds
    return summary


def _report_progress(
    progress_callback: ProgressCallback | None,
    value: float,
    description: str,
) -> None:
    """Report pipeline progress when a UI callback is available."""

    if progress_callback is None:
        return
    progress_callback(value, description)


def run_story_package_pipeline(
    image_path: str,
    api_key: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """Run the full story-to-video pipeline and save all generated artifacts."""

    source_path = Path(image_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Uploaded image not found: {source_path}")

    _report_progress(progress_callback, 0.02, "Preparing your drawing")
    _clear_previous_runs()
    run_dir = _new_run_dir()
    copied_input_path = run_dir / f"input_image{source_path.suffix.lower() or '.png'}"
    working_image_path = run_dir / "working_image.png"
    debug_print(
        "PIPELINE START",
        {
            "source_image": summarize_image(source_path),
            "run_dir": str(run_dir.resolve()),
            "story_model": os.getenv("STORYAI_STORY_MODEL", "gpt-5.4"),
            "timing_model": os.getenv("STORYAI_TRANSCRIPTION_MODEL", "whisper-1"),
            "api_key": mask_api_key(api_key or os.getenv("OPENAI_API_KEY")),
        },
    )

    shutil.copy2(source_path, copied_input_path)
    _report_progress(progress_callback, 0.08, "Normalizing the drawing")
    normalize_image(copied_input_path, working_image_path)
    debug_print("COPIED INPUT IMAGE", summarize_image(copied_input_path))
    debug_print("NORMALIZED WORKING IMAGE", summarize_image(working_image_path))

    client = build_client(api_key=api_key)
    _report_progress(progress_callback, 0.16, "Writing the story")
    story_package, raw_response = generate_story_package(client, working_image_path)
    page_images = _generate_page_images(
        client,
        story_package,
        run_dir,
        progress_callback=progress_callback,
    )
    page_audio = _generate_page_audio(
        client,
        story_package,
        run_dir,
        progress_callback=progress_callback,
    )
    page_timestamps = _generate_page_timestamps(
        client,
        page_audio,
        run_dir,
        progress_callback=progress_callback,
    )
    page_videos, final_video_summary = _render_page_videos(
        page_images=page_images,
        page_audio=page_audio,
        page_timestamps=page_timestamps,
        run_dir=run_dir,
        progress_callback=progress_callback,
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
    _report_progress(progress_callback, 1.0, "Story video ready")

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


def _generate_page_images(
    client,
    story_package: StoryPackage,
    run_dir: Path,
    progress_callback: ProgressCallback | None = None,
) -> list[dict]:
    """Generate one page image per story part and save prompt artifacts."""

    pages = [
        ("page_1", story_package.part_1),
        ("page_2", story_package.part_2),
        ("page_3", story_package.part_3),
    ]
    generated_pages: list[dict] = []

    total_pages = len(pages)
    for index, (page_name, story_part) in enumerate(pages, start=1):
        progress_value = 0.24 + ((index - 1) / total_pages) * 0.18
        _report_progress(
            progress_callback,
            progress_value,
            f"Painting page {index} of {total_pages}",
        )
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


def _generate_page_audio(
    client,
    story_package: StoryPackage,
    run_dir: Path,
    progress_callback: ProgressCallback | None = None,
) -> list[dict]:
    """Generate one narration clip per story part and save audio artifacts."""

    pages = [
        ("page_1", story_package.part_1),
        ("page_2", story_package.part_2),
        ("page_3", story_package.part_3),
    ]
    generated_audio: list[dict] = []

    total_pages = len(pages)
    for index, (page_name, story_part) in enumerate(pages, start=1):
        progress_value = 0.46 + ((index - 1) / total_pages) * 0.14
        _report_progress(
            progress_callback,
            progress_value,
            f"Recording narration {index} of {total_pages}",
        )
        output_path = run_dir / f"{page_name}_audio.wav"
        title_text: str | None = None
        story_narration_text = _sanitize_for_narration(story_part.text)

        if page_name == "page_1":
            title_text, story_narration_text, narration_text = _build_page_one_narration(
                story_package.title,
                story_part.text,
            )
            title_audio_path = run_dir / "page_1_title_audio.wav"
            story_audio_path = run_dir / "page_1_story_audio.wav"
            title_response = synthesize_narration(
                client=client,
                text=title_text,
                output_path=title_audio_path,
            )
            story_response = synthesize_narration(
                client=client,
                text=story_narration_text,
                output_path=story_audio_path,
            )
            combined_audio = _combine_wav_with_silence(
                first_audio_path=title_audio_path,
                second_audio_path=story_audio_path,
                output_path=output_path,
            )
            response_summary = {
                "mode": "title_pause_story",
                "pause_seconds": TITLE_PAUSE_SECONDS,
                "title_audio": title_response,
                "story_audio": story_response,
                "output_audio": combined_audio,
            }
        else:
            narration_text = story_narration_text
            response_summary = synthesize_narration(
                client=client,
                text=narration_text,
                output_path=output_path,
            )

        generated_audio.append(
            {
                "page": page_name,
                "title_text": title_text,
                "story_text": story_part.text,
                "story_narration_text": story_narration_text,
                "narration_text": narration_text,
                "audio_path": str(output_path.resolve()),
                "response": response_summary,
            }
        )

    debug_print("PAGE AUDIO", generated_audio)
    return generated_audio


def _generate_page_timestamps(
    client,
    page_audio: list[dict],
    run_dir: Path,
    progress_callback: ProgressCallback | None = None,
) -> list[dict]:
    """Transcribe each narration clip and save page-level timing artifacts."""

    generated_timestamps: list[dict] = []

    total_pages = len(page_audio)
    for index, page_audio_item in enumerate(page_audio, start=1):
        progress_value = 0.62 + ((index - 1) / total_pages) * 0.14
        _report_progress(
            progress_callback,
            progress_value,
            f"Timing the words for page {index} of {total_pages}",
        )
        page_name = page_audio_item["page"]
        audio_path = Path(page_audio_item["audio_path"])
        output_path = run_dir / f"{page_name}_timestamps.json"
        raw_response = transcribe_audio_with_word_timestamps(
            client=client,
            audio_path=audio_path,
            expected_text=page_audio_item["narration_text"],
        )
        _write_json(output_path, raw_response)
        validation = raw_response.get("storyai_validation", {}) or {}

        generated_timestamps.append(
            {
                "page": page_name,
                "audio_path": str(audio_path.resolve()),
                "timestamps_path": str(output_path.resolve()),
                "story_text": page_audio_item["story_text"],
                "narration_text": page_audio_item["narration_text"],
                "text": raw_response.get("text", ""),
                "words": raw_response.get("words", []) or [],
                "segments": raw_response.get("segments", []) or [],
                "duration_seconds": raw_response.get("duration"),
                "validation": validation,
                "usable_for_word_highlight": validation.get(
                    "usable_for_word_highlight",
                    False,
                ),
                "fallback_reason": validation.get("fallback_reason"),
            }
        )

    debug_print("PAGE TIMESTAMPS", generated_timestamps)
    return generated_timestamps


def _render_page_videos(
    page_images: list[dict],
    page_audio: list[dict],
    page_timestamps: list[dict],
    run_dir: Path,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[dict], dict]:
    """Render one page clip per story part, then concatenate the final video."""

    image_by_page = {page["page"]: page for page in page_images}
    audio_by_page = {page["page"]: page for page in page_audio}
    timestamps_by_page = {page["page"]: page for page in page_timestamps}
    page_videos: list[dict] = []
    output_paths: list[Path] = []

    page_names = ("page_1", "page_2", "page_3")
    total_pages = len(page_names)
    for index, page_name in enumerate(page_names, start=1):
        progress_value = 0.78 + ((index - 1) / total_pages) * 0.16
        _report_progress(
            progress_callback,
            progress_value,
            f"Rendering video page {index} of {total_pages}",
        )
        image_item = image_by_page[page_name]
        audio_item = audio_by_page[page_name]
        timestamp_item = timestamps_by_page[page_name]
        output_path = run_dir / f"{page_name}.mp4"
        audio_duration = (
            audio_item.get("response", {})
            .get("output_audio", {})
            .get("duration_seconds")
        )
        render_summary = render_page_video(
            image_path=Path(image_item["image_path"]),
            audio_path=Path(audio_item["audio_path"]),
            story_text=audio_item["story_narration_text"],
            title_text=audio_item.get("title_text"),
            words=timestamp_item["words"],
            duration_seconds=float(timestamp_item["duration_seconds"] or audio_duration or 0.0),
            output_path=output_path,
        )
        page_videos.append(
            {
                "page": page_name,
                "title_text": audio_item.get("title_text"),
                "story_text": audio_item["story_text"],
                "narration_text": audio_item["narration_text"],
                "image_path": image_item["image_path"],
                "audio_path": audio_item["audio_path"],
                "timestamps_path": timestamp_item["timestamps_path"],
                "subtitle_script_path": render_summary["subtitle_script_path"],
                "video_path": render_summary["video_path"],
                "subtitle_mode": render_summary["response"]["subtitle_mode"],
                "fallback_reason": render_summary["response"].get("fallback_reason"),
                "response": render_summary["response"],
            }
        )
        output_paths.append(output_path)

    debug_print("PAGE VIDEOS", page_videos)

    final_output_path = run_dir / "final_story.mp4"
    _report_progress(progress_callback, 0.96, "Stitching the final story video")
    final_video_response = concatenate_page_videos(output_paths, final_output_path)
    final_video_summary = {
        "video_path": str(final_output_path.resolve()),
        "response": final_video_response,
    }
    debug_print("FINAL VIDEO", final_video_summary)
    return page_videos, final_video_summary

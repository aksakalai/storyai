import base64
import json
import mimetypes
import os
import re
import unicodedata
from pathlib import Path

from openai import OpenAI

from .debug_utils import debug_print, summarize_image, summarize_path, summarize_wav_audio
from .prompts import (
    NARRATION_INSTRUCTIONS,
    SYSTEM_PROMPT,
    USER_PROMPT,
    build_page_image_prompt,
)
from .schemas import StoryPackage


DEFAULT_STORY_MODEL = os.getenv("STORYAI_STORY_MODEL", "gpt-5.4")
DEFAULT_IMAGE_DETAIL = os.getenv("STORYAI_IMAGE_DETAIL", "low")
DEFAULT_PAGE_IMAGE_SIZE = os.getenv("STORYAI_PAGE_IMAGE_SIZE", "1024x1024")
DEFAULT_PAGE_IMAGE_MODEL = os.getenv("STORYAI_PAGE_IMAGE_MODEL", "gpt-image-1.5")
DEFAULT_PAGE_IMAGE_QUALITY = os.getenv("STORYAI_PAGE_IMAGE_QUALITY", "low")
MEDIUM_QUALITY_PAGE_IMAGE_MODEL = os.getenv(
    "STORYAI_MEDIUM_QUALITY_PAGE_IMAGE_MODEL",
    DEFAULT_PAGE_IMAGE_MODEL,
)
MEDIUM_QUALITY_PAGE_IMAGE_QUALITY = os.getenv(
    "STORYAI_MEDIUM_QUALITY_PAGE_IMAGE_QUALITY",
    "medium",
)
HIGH_QUALITY_PAGE_IMAGE_MODEL = os.getenv(
    "STORYAI_HIGH_QUALITY_PAGE_IMAGE_MODEL",
    DEFAULT_PAGE_IMAGE_MODEL,
)
HIGH_QUALITY_PAGE_IMAGE_QUALITY = os.getenv(
    "STORYAI_HIGH_QUALITY_PAGE_IMAGE_QUALITY",
    "high",
)
DEFAULT_TTS_MODEL = os.getenv("STORYAI_TTS_MODEL", "gpt-4o-mini-tts")
DEFAULT_TTS_VOICE = os.getenv("STORYAI_TTS_VOICE", "marin")
DEFAULT_TTS_FORMAT = os.getenv("STORYAI_TTS_FORMAT", "wav")
DEFAULT_TRANSCRIPTION_MODEL = os.getenv("STORYAI_TRANSCRIPTION_MODEL", "whisper-1")
DEFAULT_TRANSCRIPTION_RESPONSE_FORMAT = os.getenv(
    "STORYAI_TRANSCRIPTION_RESPONSE_FORMAT",
    "verbose_json",
)
DEFAULT_TRANSCRIPTION_LANGUAGE = os.getenv("STORYAI_TRANSCRIPTION_LANGUAGE", "en")


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a simple true/false style environment flag."""

    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_image_mode() -> str:
    """Resolve the simple image mode selector used by Colab and local runs."""

    raw_mode = os.getenv("STORYAI_IMAGE_MODE")
    if raw_mode:
        normalized = raw_mode.strip().lower()
        if normalized in {"0", "default", "budget", "low", "false"}:
            return "default"
        if normalized in {"1", "medium", "med"}:
            return "medium"
        if normalized in {"2", "high", "hq", "true"}:
            return "high"

    if _env_flag("STORYAI_HIGH_QUALITY_IMAGES", default=False):
        return "high"
    return "default"


def resolve_page_image_settings(
    model: str | None = None,
    quality: str | None = None,
) -> tuple[str, str]:
    """Resolve image generation settings from a simple quality flag or explicit overrides."""

    image_mode = resolve_image_mode()
    if image_mode == "medium":
        profile_model = MEDIUM_QUALITY_PAGE_IMAGE_MODEL
        profile_quality = MEDIUM_QUALITY_PAGE_IMAGE_QUALITY
    elif image_mode == "high":
        profile_model = HIGH_QUALITY_PAGE_IMAGE_MODEL
        profile_quality = HIGH_QUALITY_PAGE_IMAGE_QUALITY
    else:
        profile_model = DEFAULT_PAGE_IMAGE_MODEL
        profile_quality = DEFAULT_PAGE_IMAGE_QUALITY

    resolved_model = model or os.getenv("STORYAI_PAGE_IMAGE_MODEL") or profile_model
    resolved_quality = (
        quality or os.getenv("STORYAI_PAGE_IMAGE_QUALITY") or profile_quality
    )
    return resolved_model, resolved_quality


def get_runtime_model_config() -> dict[str, str | bool]:
    """Return the effective model configuration for logging and user guidance."""

    image_model, image_quality = resolve_page_image_settings()
    image_mode = resolve_image_mode()

    return {
        "story_model": DEFAULT_STORY_MODEL,
        "image_model": image_model,
        "image_quality": image_quality,
        "image_mode": image_mode,
        "high_quality_images": image_mode == "high",
        "tts_model": DEFAULT_TTS_MODEL,
        "timing_model": DEFAULT_TRANSCRIPTION_MODEL,
    }


def build_client(api_key: str | None = None) -> OpenAI:
    """Create an OpenAI client from an explicit key or environment variable."""

    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=resolved_key)


def image_to_data_url(image_path: Path) -> str:
    """Convert a local image file to a data URL for the Responses API."""

    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def generate_story_package(
    client: OpenAI,
    image_path: Path,
    model: str = DEFAULT_STORY_MODEL,
) -> tuple[StoryPackage, dict]:
    """Generate a structured story package from a normalized image."""

    request_payload = {
        "model": model,
        "image_detail": DEFAULT_IMAGE_DETAIL,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": USER_PROMPT,
        "image": summarize_image(image_path),
        "response_format": StoryPackage.model_json_schema(),
    }
    debug_print("OPENAI REQUEST", request_payload)

    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": USER_PROMPT,
                    },
                    {
                        "type": "input_image",
                        "image_url": image_to_data_url(image_path),
                        "detail": DEFAULT_IMAGE_DETAIL,
                    },
                ],
            },
        ],
        text_format=StoryPackage,
    )

    story_package = response.output_parsed
    if story_package is None:
        raise RuntimeError("The model response could not be parsed into StoryPackage.")

    raw_response = response.model_dump(mode="json", warnings=False)

    debug_print(
        "OPENAI RESPONSE SUMMARY",
        {
            "response_id": raw_response.get("id"),
            "model": raw_response.get("model"),
            "usage": raw_response.get("usage"),
            "status": raw_response.get("status"),
        },
    )
    debug_print("PARSED STORY PACKAGE", story_package.model_dump(mode="json"))
    debug_print("RAW OPENAI RESPONSE", raw_response)

    return story_package, raw_response


def generate_page_image(
    client: OpenAI,
    visual_canon: str,
    page_prompt: str,
    output_path: Path,
    model: str | None = None,
    size: str = DEFAULT_PAGE_IMAGE_SIZE,
    quality: str | None = None,
) -> tuple[str, dict]:
    """Generate one page image and save it locally."""

    resolved_model, resolved_quality = resolve_page_image_settings(
        model=model,
        quality=quality,
    )
    final_prompt = build_page_image_prompt(visual_canon, page_prompt)
    request_payload = {
        "model": resolved_model,
        "size": size,
        "quality": resolved_quality,
        "output_path": str(output_path.resolve()),
        "prompt": final_prompt,
    }
    debug_print("IMAGE GENERATION REQUEST", request_payload)

    response = client.images.generate(
        model=resolved_model,
        prompt=final_prompt,
        size=size,
        quality=resolved_quality,
    )

    image_data = response.data[0]
    image_base64 = getattr(image_data, "b64_json", None)
    if not image_base64 and hasattr(image_data, "model_dump"):
        image_base64 = image_data.model_dump().get("b64_json")
    if not image_base64:
        raise RuntimeError("Image generation response did not include base64 image data.")

    output_path.write_bytes(base64.b64decode(image_base64))

    response_summary = {
        "created": getattr(response, "created", None),
        "size": size,
        "quality": resolved_quality,
        "model": resolved_model,
        "output_image": summarize_path(output_path),
    }
    debug_print("IMAGE GENERATION RESPONSE", response_summary)

    return final_prompt, response_summary


def synthesize_narration(
    client: OpenAI,
    text: str,
    output_path: Path,
    model: str = DEFAULT_TTS_MODEL,
    voice: str = DEFAULT_TTS_VOICE,
    response_format: str = DEFAULT_TTS_FORMAT,
    instructions: str = NARRATION_INSTRUCTIONS,
) -> dict:
    """Generate one narration audio file and save it locally."""

    request_payload = {
        "model": model,
        "voice": voice,
        "response_format": response_format,
        "output_path": str(output_path.resolve()),
        "instructions": instructions.strip(),
        "text": text,
    }
    debug_print("TTS REQUEST", request_payload)

    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text,
        instructions=instructions,
        response_format=response_format,
    ) as response:
        response.stream_to_file(output_path)

    response_summary = {
        "model": model,
        "voice": voice,
        "response_format": response_format,
        "instructions": instructions.strip(),
        "output_audio": summarize_wav_audio(output_path),
    }
    debug_print("TTS RESPONSE", response_summary)

    return response_summary


def transcribe_audio_with_word_timestamps(
    client: OpenAI,
    audio_path: Path,
    expected_text: str,
    model: str = DEFAULT_TRANSCRIPTION_MODEL,
    response_format: str = DEFAULT_TRANSCRIPTION_RESPONSE_FORMAT,
    language: str = DEFAULT_TRANSCRIPTION_LANGUAGE,
) -> dict:
    """Transcribe one narration clip with strict word-level timestamps."""

    request_payload = {
        "model": model,
        "response_format": response_format,
        "timestamp_granularities": ["word"],
        "language": language,
        "audio": summarize_wav_audio(audio_path),
        "expected_text": expected_text,
    }
    debug_print("TRANSCRIPTION REQUEST", request_payload)

    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            file=audio_file,
            model=model,
            response_format=response_format,
            timestamp_granularities=["word"],
            language=language,
            prompt=expected_text,
        )

    raw_response = _to_jsonable(response)
    _validate_transcription(raw_response, expected_text=expected_text)

    debug_print(
        "TRANSCRIPTION RESPONSE SUMMARY",
        {
            "model": model,
            "response_format": response_format,
            "language": raw_response.get("language"),
            "duration": raw_response.get("duration"),
            "text": raw_response.get("text"),
            "word_count": len(raw_response.get("words", []) or []),
            "segment_count": len(raw_response.get("segments", []) or []),
        },
    )
    debug_print("RAW TRANSCRIPTION RESPONSE", raw_response)

    return raw_response


def _validate_transcription(raw_response: dict, expected_text: str) -> None:
    """Require usable word timings and an exact normalized text match."""

    words = raw_response.get("words", []) or []
    if not words:
        raise RuntimeError("Transcription did not return any word timestamps.")

    expected_tokens = _normalize_transcript_tokens(expected_text)
    if not expected_tokens:
        raise RuntimeError("Story text did not contain any tokens for transcription validation.")

    transcript_text = str(raw_response.get("text", "") or "")
    transcript_tokens = _normalize_transcript_tokens(transcript_text)
    word_tokens = _normalize_transcript_tokens(
        " ".join(str(word.get("word", "") or "") for word in words)
    )

    if word_tokens != expected_tokens and transcript_tokens != expected_tokens:
        raise RuntimeError("Transcription did not match the expected story text exactly.")


def _normalize_transcript_tokens(text: str) -> list[str]:
    """Normalize text into lowercase word tokens for strict transcript checks."""

    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = normalized.replace("\u2019", "'")
    normalized = normalized.replace("\u2018", "'")
    normalized = normalized.replace("\u201c", '"')
    normalized = normalized.replace("\u201d", '"')
    normalized = normalized.replace("\u2014", " ")
    normalized = normalized.replace("\u2013", " ")
    normalized = normalized.replace("-", " ")

    tokens: list[str] = []
    for raw_token in normalized.split():
        cleaned = re.sub(r"^[^\w']+|[^\w']+$", "", raw_token)
        cleaned = re.sub(r"[^a-z0-9']+", "", cleaned)
        if cleaned:
            tokens.append(cleaned)
    return tokens


def _to_jsonable(value) -> dict:
    """Convert SDK response objects into a plain JSON-serializable dict."""

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", warnings=False)
    if isinstance(value, dict):
        return value
    return json.loads(json.dumps(value, default=str))

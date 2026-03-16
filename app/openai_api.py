import base64
import mimetypes
import os
from pathlib import Path

from openai import OpenAI

from .debug_utils import debug_print, summarize_image
from .prompts import SYSTEM_PROMPT, USER_PROMPT
from .schemas import StoryPackage


DEFAULT_STORY_MODEL = os.getenv("STORYAI_STORY_MODEL", "gpt-4o-mini")
DEFAULT_IMAGE_DETAIL = os.getenv("STORYAI_IMAGE_DETAIL", "low")


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

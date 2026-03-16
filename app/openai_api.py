import base64
import mimetypes
import os
from pathlib import Path

from openai import OpenAI

from .debug_utils import debug_print, summarize_image, summarize_path
from .prompts import SYSTEM_PROMPT, USER_PROMPT, build_page_image_prompt
from .schemas import StoryPackage


DEFAULT_STORY_MODEL = os.getenv("STORYAI_STORY_MODEL", "gpt-4o-mini")
DEFAULT_IMAGE_DETAIL = os.getenv("STORYAI_IMAGE_DETAIL", "low")
DEFAULT_PAGE_IMAGE_MODEL = os.getenv("STORYAI_PAGE_IMAGE_MODEL", "gpt-image-1-mini")
DEFAULT_PAGE_IMAGE_SIZE = os.getenv("STORYAI_PAGE_IMAGE_SIZE", "1024x1024")
DEFAULT_PAGE_IMAGE_QUALITY = os.getenv("STORYAI_PAGE_IMAGE_QUALITY", "low")


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
    model: str = DEFAULT_PAGE_IMAGE_MODEL,
    size: str = DEFAULT_PAGE_IMAGE_SIZE,
    quality: str = DEFAULT_PAGE_IMAGE_QUALITY,
) -> tuple[str, dict]:
    """Generate one page image and save it locally."""

    final_prompt = build_page_image_prompt(visual_canon, page_prompt)
    request_payload = {
        "model": model,
        "size": size,
        "quality": quality,
        "output_path": str(output_path.resolve()),
        "prompt": final_prompt,
    }
    debug_print("IMAGE GENERATION REQUEST", request_payload)

    response = client.images.generate(
        model=model,
        prompt=final_prompt,
        size=size,
        quality=quality,
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
        "quality": quality,
        "model": model,
        "output_image": summarize_path(output_path),
    }
    debug_print("IMAGE GENERATION RESPONSE", response_summary)

    return final_prompt, response_summary

SYSTEM_PROMPT = """\
You create warm, child-friendly bedtime stories inspired by a child's drawing.
Return exactly the requested schema.

Story rules:
- Build a complete three-part arc.
- part_1 is the setup.
- part_2 introduces a gentle change or problem.
- part_3 resolves the story in a calm, comforting way.
- Keep the tone soft, safe, and bedtime-friendly.
- Keep visual_canon lightweight and practical for consistent illustrations.
- Do not use copyrighted characters or brand names.
"""


USER_PROMPT = """\
Use the uploaded drawing as inspiration and return one StoryPackage object.

Requirements:
- Include title and visual_canon.
- Include part_1, part_2, and part_3.
- Each part must contain text and image_prompt.
- Each text section should be short enough for one narrated story page.
- Each image_prompt should stay visually consistent with the same characters, colors, and world.
"""


PAGE_IMAGE_PROMPT_TEMPLATE = """\
Create one finished bedtime-story illustration for a single page.

Shared visual canon for consistency:
{visual_canon}

Specific scene for this page:
{page_prompt}

Requirements:
- Keep the same characters, colors, setting, and mood consistent with the shared visual canon.
- Make the image warm, child-friendly, and suitable for a bedtime storybook.
- Show only the illustrated scene for this page.
- Do not add text, letters, captions, speech bubbles, page numbers, borders, or watermarks.
"""


def build_page_image_prompt(visual_canon: str, page_prompt: str) -> str:
    """Build the final prompt used for page image generation."""

    return PAGE_IMAGE_PROMPT_TEMPLATE.format(
        visual_canon=visual_canon.strip(),
        page_prompt=page_prompt.strip(),
    )

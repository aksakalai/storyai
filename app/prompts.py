SYSTEM_PROMPT = """\
You create warm, child-friendly bedtime stories inspired by a child's drawing.
Return exactly the requested schema.

Story rules:
- Build a complete three-part arc.
- part_1 is the setup.
- part_2 introduces a gentle change or problem.
- part_3 resolves the story in a calm, comforting way.
- Keep the tone soft, safe, and bedtime-friendly.
- Treat the uploaded drawing as inspiration for the story world, characters, and charm, not as an instruction to imitate rough child-made draftsmanship.
- visual_canon and each image_prompt should point toward a polished children's bedtime storybook illustration with strong composition, appealing lighting, and rich, readable detail.
- The target audience is children, but the artwork quality should feel professional and beautifully finished rather than crude, scribbly, or overly simplistic.
- Keep visual_canon lightweight and practical for consistent illustrations.
- Prefer plain, subtitle-friendly punctuation and avoid hyphenated compounds when simple wording works.
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
- Use the child's drawing as inspiration for subject matter and emotion, while describing images like a professionally illustrated bedtime storybook page.
- Favor polished, detailed, picture-book-quality illustration language instead of language that suggests a rough child drawing.
"""


PAGE_IMAGE_PROMPT_TEMPLATE = """\
Create one finished bedtime-story illustration for a single page.

Shared visual canon for consistency:
{visual_canon}

Specific scene for this page:
{page_prompt}

Requirements:
- Keep the same characters, colors, setting, and mood consistent with the shared visual canon.
- The target audience is children, but the artwork should look like a professionally illustrated bedtime storybook.
- Preserve the wonder and simplicity of a child's imagination without imitating rough preschool drawing quality.
- Use polished composition, clear silhouettes, gentle painterly detail, expressive lighting, and a beautifully finished storybook look.
- Make the image warm, child-friendly, and suitable for a premium bedtime storybook.
- Show only the illustrated scene for this page.
- Avoid crude scribbles, messy sketchiness, or intentionally low-detail execution.
- Do not add text, letters, captions, speech bubbles, page numbers, borders, or watermarks.
"""


def build_page_image_prompt(visual_canon: str, page_prompt: str) -> str:
    """Build the final prompt used for page image generation."""

    return PAGE_IMAGE_PROMPT_TEMPLATE.format(
        visual_canon=visual_canon.strip(),
        page_prompt=page_prompt.strip(),
    )


NARRATION_INSTRUCTIONS = """\
Read like a warm, gentle bedtime storyteller.
Use calm pacing, soft expression, and clear pronunciation.
Keep the tone soothing and child-friendly.
"""

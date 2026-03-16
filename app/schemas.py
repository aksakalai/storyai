from pydantic import BaseModel, Field


class StoryPart(BaseModel):
    """Narration and illustration prompt for a single story page."""

    text: str = Field(min_length=1)
    image_prompt: str = Field(min_length=1)


class StoryPackage(BaseModel):
    """Structured response returned from the single story generation call."""

    title: str = Field(min_length=1)
    visual_canon: str = Field(min_length=1)
    part_1: StoryPart
    part_2: StoryPart
    part_3: StoryPart

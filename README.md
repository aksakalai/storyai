# StoryAI

StoryAI is a Colab-first Gradio app for turning a single child drawing into a structured bedtime story package with OpenAI.

The current scaffold implements the first phase of the agreed workflow:

- upload or capture one drawing
- save the original image and a normalized working copy
- make one OpenAI call for a structured `StoryPackage`
- save artifacts locally
- show the story parts in the Gradio UI

The later phases for page image generation, narration, timestamps, subtitle rendering, and final video assembly will build on top of this repo structure.

## Colab

Run the whole app from one Colab cell:

```python
!rm -rf /content/storyai
!git clone https://github.com/aksakalai/storyai.git /content/storyai
%cd /content/storyai
!python -m pip install -U pip
!pip install -r requirements.txt

import os

os.environ["OPENAI_API_KEY"] = "PASTE_YOUR_KEY_HERE"
os.environ["STORYAI_SHARE"] = "true"

!python -u -m app.main
```

This keeps the repo clean while still letting you paste the API key directly into the Colab cell for private demo use.

## Local layout

```text
app/
  main.py
  openai_api.py
  pipeline.py
  prompts.py
  schemas.py
requirements.txt
README.md
```

## Artifacts

By default, the app stores artifacts in `runs/<timestamp>_<id>/`.

Each run currently writes:

- `input_image.<ext>`
- `working_image.png`
- `story_package.json`
- `openai_response.json`

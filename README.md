# StoryAI

StoryAI is a Colab-first Gradio app for turning a single child drawing into a structured bedtime story package with OpenAI.

The current scaffold implements the first two phases of the agreed workflow:

- upload or capture one drawing
- save the original image and a normalized working copy
- make one OpenAI call for a structured `StoryPackage`
- generate 3 page images from the story package
- save artifacts locally
- show the story parts, generated images, and final image prompts in the Gradio UI

The later phases for narration, timestamps, subtitle rendering, and final video assembly will build on top of this repo structure.

## Colab

Run the whole app from one Colab cell:

```python
import os
import shutil
import subprocess
import sys

os.chdir("/content")
shutil.rmtree("/content/storyai", ignore_errors=True)
subprocess.run(
    ["git", "clone", "https://github.com/aksakalai/storyai.git", "/content/storyai"],
    check=True,
)
os.chdir("/content/storyai")
subprocess.run([sys.executable, "-m", "pip", "install", "-U", "pip"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

os.environ["OPENAI_API_KEY"] = "PASTE_YOUR_KEY_HERE"
os.environ["STORYAI_SHARE"] = "true"

subprocess.run([sys.executable, "-u", "-m", "app.main"], check=True)
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
- `page_1.png`
- `page_2.png`
- `page_3.png`
- `page_1_prompt.txt`
- `page_2_prompt.txt`
- `page_3_prompt.txt`
- `page_images.json`

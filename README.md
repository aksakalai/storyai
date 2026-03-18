# StoryAI

StoryAI is a Colab-first Gradio app for turning a single child drawing into a structured bedtime story package with OpenAI.

The current scaffold implements the agreed single-run demo workflow:

- upload or capture one drawing
- save the original image and a normalized working copy
- make one OpenAI call for a structured `StoryPackage`
- generate 3 page images from the story package
- generate 3 narration audio clips from the story parts
- transcribe the 3 narration clips with word timestamps
- render 3 page video clips with persistent word highlighting
- concatenate the 3 clips into a final story video
- save artifacts locally
- show the story parts, generated images, narration audio, transcripts, word timings, and final stitched video in the Gradio UI

This now covers the full single-run demo pipeline on Colab CPU.

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
subprocess.run(["apt-get", "update"], check=True)
subprocess.run(["apt-get", "install", "-y", "ffmpeg"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-U", "pip"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

os.environ["OPENAI_API_KEY"] = "PASTE_YOUR_KEY_HERE"
os.environ["STORYAI_SHARE"] = "true"

# Image quality switch for non-coders:
# - "false" = Default / budget-friendly mode
#   Uses GPT Image 1.5 at low quality.
#   Roughly about $0.06 per full story run.
#   With a balance of about $9.80, that is roughly about 150 full runs.
#
# - "true" = High-quality mode
#   Uses GPT Image 1.5 at high quality.
#   Roughly about $0.43 per full story run.
#   With a balance of about $9.80, that is roughly about 22 full runs.
#
# Only the image generation changes between these two modes.
# The story model, narration model, and transcription model stay fixed.
# Use high-quality mode when presenting the project live or recording a demo.
# Use the default mode for normal testing so the balance lasts much longer.
os.environ["STORYAI_HIGH_QUALITY_IMAGES"] = "false"

subprocess.run([sys.executable, "-u", "-m", "app.main"], check=True)
```

This keeps the repo clean while still letting you paste the API key directly into the Colab cell for private demo use.

### Simple quality switch

StoryAI now uses these fixed models:

- story: `gpt-5.4`
- narration: `gpt-4o-mini-tts`
- transcription: `gpt-4o-mini-transcribe`

The only simple switch is image generation:

- Default / budget-friendly mode: `STORYAI_HIGH_QUALITY_IMAGES="false"`
  - image model: `gpt-image-1.5`
  - image quality: `low`
  - rough cost: about `$0.06` per full story run
  - rough run count from about `$9.80` balance: about `150` runs
- High-quality mode: `STORYAI_HIGH_QUALITY_IMAGES="true"`
  - image model: `gpt-image-1.5`
  - image quality: `high`
  - rough cost: about `$0.43` per full story run
  - rough run count from about `$9.80` balance: about `22` runs

These are rough planning numbers based on the current OpenAI pricing as of March 18, 2026 and a typical StoryAI run with about 1.5 minutes of total narration. Longer stories will cost a bit more. If you are getting close to those run counts, top up the balance before continuing.

## Local layout

```text
app/
  main.py
  openai_api.py
  pipeline.py
  prompts.py
  schemas.py
  video.py
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
- `page_1_audio.wav`
- `page_2_audio.wav`
- `page_3_audio.wav`
- `page_1_timestamps.json`
- `page_2_timestamps.json`
- `page_3_timestamps.json`
- `page_1_subtitles.ass`
- `page_2_subtitles.ass`
- `page_3_subtitles.ass`
- `page_1_prompt.txt`
- `page_2_prompt.txt`
- `page_3_prompt.txt`
- `page_1.mp4`
- `page_2.mp4`
- `page_3.mp4`
- `page_videos_concat.txt`
- `page_images.json`
- `page_audio.json`
- `page_timestamps.json`
- `page_videos.json`
- `final_story.mp4`

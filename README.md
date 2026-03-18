# StoryAI

StoryAI is a Colab-first Gradio app for turning a single child drawing into a structured bedtime story package with OpenAI.

The current scaffold implements the agreed single-run demo workflow:

- upload or capture one drawing
- save the original image and a normalized working copy
- make one OpenAI call for a structured `StoryPackage`
- generate 3 page images from the story package
- generate 3 narration audio clips from the story parts
- align the 3 narration clips to the exact story text with a local word aligner
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
from importlib import metadata

os.chdir("/content")
shutil.rmtree("/content/storyai", ignore_errors=True)
!git clone https://github.com/aksakalai/storyai.git /content/storyai
os.chdir("/content/storyai")
!apt-get update
!apt-get install -y ffmpeg
!pip install -U pip

def version_or_none(package_name):
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None

torch_version = version_or_none("torch")
torchaudio_version = version_or_none("torchaudio")
needs_alignment_runtime = (
    torch_version != "2.8.0" or torchaudio_version != "2.8.0"
)

print("torch:", torch_version)
print("torchaudio:", torchaudio_version)
print("install alignment runtime:", needs_alignment_runtime)

if needs_alignment_runtime:
    get_ipython().system(
        "pip install --index-url https://download.pytorch.org/whl/cpu "
        "torch==2.8.0 torchaudio==2.8.0"
    )

!pip install -r requirements.txt

os.environ["OPENAI_API_KEY"] = "PASTE_YOUR_KEY_HERE"
os.environ["STORYAI_SHARE"] = "true"

# Image quality switch for non-coders:
# - "default" = Default / budget-friendly mode
#   Uses GPT Image 1.5 at low quality.
#   Roughly about $0.06 per full story run.
#   With a balance of about $9.80, that is roughly about 150 full runs.
#
# - "medium" = Medium quality mode
#   Uses GPT Image 1.5 at medium quality.
#   Roughly about $0.14 per full story run.
#   With a balance of about $9.80, that is roughly about 70 full runs.
#
# - "high" = High-quality mode
#   Uses GPT Image 1.5 at high quality.
#   Roughly about $0.43 per full story run.
#   With a balance of about $9.80, that is roughly about 22 full runs.
#
# Only the image generation changes between these modes.
# The story model and narration model stay fixed.
# Word timing is generated locally for free.
# Use high-quality mode when presenting the project live or recording a demo.
# Use the default mode for normal testing so the balance lasts much longer.
# If you are getting close to those run counts, tell Baris to top up the balance.
os.environ["STORYAI_IMAGE_MODE"] = "default"

!python -u -m app.main
```

This keeps the repo clean while still letting you paste the API key directly into the Colab cell for private demo use.
The first generation in a fresh Colab runtime will also download the local alignment model once, so that run will start more slowly than the next ones.
The PyTorch install step checks the current versions first, then only installs the CPU-only wheels if the runtime is missing the compatible `2.8.0` pair.

### Simple quality switch

StoryAI now uses these fixed models:

- story: `gpt-5.4`
- narration: `gpt-4o-mini-tts`
- timing: local forced alignment with `WAV2VEC2_ASR_BASE_960H`

The only simple switch is image generation:

- Default / budget-friendly mode: `STORYAI_IMAGE_MODE="default"`
  - image model: `gpt-image-1.5`
  - image quality: `low`
  - rough cost: about `$0.06` per full story run
  - rough run count from about `$9.80` balance: about `150` runs
- Medium quality mode: `STORYAI_IMAGE_MODE="medium"`
  - image model: `gpt-image-1.5`
  - image quality: `medium`
  - rough cost: about `$0.14` per full story run
  - rough run count from about `$9.80` balance: about `70` runs
- High-quality mode: `STORYAI_IMAGE_MODE="high"`
  - image model: `gpt-image-1.5`
  - image quality: `high`
  - rough cost: about `$0.43` per full story run
  - rough run count from about `$9.80` balance: about `22` runs

These are rough planning numbers based on the current OpenAI pricing as of March 18, 2026 and a typical StoryAI run with about 1.5 minutes of total narration. The timing step is local and free, so the main cost difference is still the image mode. If you are getting close to those run counts, top up the balance before continuing.

## Local layout

```text
app/
  alignment.py
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

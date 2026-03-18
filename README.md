# StoryAI

StoryAI is a Colab-first Gradio app that turns one child drawing into a three-page bedtime story video with OpenAI.

The interface has two modes:

- Child Mode starts in a simple upload-first view and generates the story video automatically after an image is added.
- Parent Mode exposes the full debugger-style page with the working image, prompts, narration, timestamps, and saved artifact paths.

The app does one full run:

- upload or capture one drawing
- save the original image and a normalized working copy
- generate one structured `StoryPackage`
- generate 3 page images
- generate 3 narration clips
- transcribe each narration clip with `whisper-1` word timestamps
- render 3 page videos with persistent word highlighting
- concatenate the 3 clips into one final story video
- save all artifacts locally
- show the story, images, audio, timestamps, and final video in the Gradio UI

## Colab

Run the whole app from one Colab cell:

```python
import os
import shutil

os.chdir("/content")
shutil.rmtree("/content/storyai", ignore_errors=True)
!git clone https://github.com/aksakalai/storyai.git /content/storyai

!apt-get update
!apt-get install -y ffmpeg

os.chdir("/content/storyai")
!pip install -U pip
!pip install -r requirements.txt

os.environ["OPENAI_API_KEY"] = "PASTE_YOUR_KEY_HERE"
os.environ["STORYAI_SHARE"] = "true"
os.environ["STORYAI_IMAGE_MODE"] = "default"

!python -u -m app.main
```

## Models

StoryAI uses these defaults:

- story: `gpt-5.4`
- narration: `gpt-4o-mini-tts`
- timing: `whisper-1`

The simple image switch is still:

- `STORYAI_IMAGE_MODE="default"` for low image quality
- `STORYAI_IMAGE_MODE="medium"` for medium image quality
- `STORYAI_IMAGE_MODE="high"` for high image quality

## Local Layout

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

Each run writes:

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

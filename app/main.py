import os
import threading
import traceback

import gradio as gr

from .debug_utils import DEBUG_ENABLED, debug_print
from .openai_api import get_runtime_model_config
from .pipeline import DEFAULT_RUNS_DIR, run_story_package_pipeline


RUN_LOCK = threading.Lock()
APP_CSS = """
:root {
    --storyai-page-max-width: 1220px;
    --storyai-copy-max-width: 780px;
    --storyai-panel-max-width: 1120px;
    --storyai-media-max-width: 640px;
    --storyai-control-max-width: 720px;
}

.gradio-container {
    background:
        radial-gradient(circle at top, rgba(255, 141, 54, 0.10), transparent 32%),
        linear-gradient(180deg, rgba(255, 249, 244, 0.96), rgba(255, 255, 255, 1) 18%);
}

#storyai-shell {
    margin: 0 auto;
    max-width: var(--storyai-page-max-width);
    padding: 0.75rem 0 2.5rem;
}

#storyai-hero {
    margin: 0 auto 1.5rem auto;
    max-width: var(--storyai-copy-max-width);
    text-align: center;
}

#storyai-hero h1 {
    font-size: clamp(2.75rem, 5vw, 4rem);
    line-height: 1.05;
    margin: 0;
}

#storyai-hero .storyai-subtitle {
    letter-spacing: 0.08em;
    margin-top: 0.4rem;
    opacity: 0.78;
    text-transform: uppercase;
}

#storyai-hero .storyai-intro {
    margin-top: 0.9rem;
}

#storyai-tabs [role="tablist"],
.storyai-page-tabs [role="tablist"] {
    justify-content: center;
    gap: 0.5rem;
}

#storyai-tabs [role="tab"],
.storyai-page-tabs [role="tab"] {
    flex: 0 0 auto;
}

.storyai-mode-panel {
    margin: 0 auto;
    max-width: var(--storyai-panel-max-width);
}

.storyai-tab-copy {
    margin: 0 auto 1rem auto;
    max-width: var(--storyai-copy-max-width);
    text-align: center;
}

.storyai-tab-copy p {
    margin: 0;
}

.storyai-media-row {
    justify-content: center;
    gap: 1.5rem;
}

.storyai-media {
    margin-left: auto !important;
    margin-right: auto !important;
    max-width: var(--storyai-media-max-width);
    width: 100%;
}

.storyai-media img,
.storyai-media video,
.storyai-media canvas,
.storyai-media .wrap,
.storyai-media .empty-state,
.storyai-media .image-container,
.storyai-media .video-container {
    max-width: 100%;
}

.storyai-controls {
    margin: 1.1rem auto 1.6rem auto;
    max-width: var(--storyai-control-max-width);
    gap: 0.8rem;
}

.storyai-action {
    margin-left: auto !important;
    margin-right: auto !important;
    max-width: var(--storyai-control-max-width);
    width: 100%;
}

.storyai-action button {
    border-radius: 999px !important;
    box-shadow: 0 12px 30px rgba(255, 122, 26, 0.18) !important;
}
"""


def _report_gradio_progress(
    progress: gr.Progress | None,
    value: float,
    description: str,
) -> None:
    """Update the Gradio progress bar when a run is active."""

    if progress is None:
        return
    progress(value, desc=description)


def _run_generation(
    image_path: str,
    progress: gr.Progress | None = None,
    source: str = "ui",
) -> dict:
    """Generate the full story, media, and rendered video outputs."""

    if not image_path:
        raise gr.Error("Upload or capture one drawing first.")

    if not RUN_LOCK.acquire(blocking=False):
        raise gr.Error("A run is already in progress. Please wait for it to finish.")

    try:
        debug_print(
            "GRADIO EVENT",
            {
                "input_image_path": image_path,
                "source": source,
            },
        )
        _report_gradio_progress(progress, 0.0, "Starting StoryAI")
        result = run_story_package_pipeline(
            image_path,
            progress_callback=lambda value, description: _report_gradio_progress(
                progress,
                value,
                description,
            ),
        )
        story = result["story_package"]
        debug_print(
            "GRADIO RESULT",
            {
                "title": story.title,
                "run_dir": result["run_dir"],
                "working_image": result["working_image"],
                "story_package_path": result["story_package_path"],
                "openai_response_path": result["openai_response_path"],
                "page_image_manifest_path": result["page_image_manifest_path"],
                "page_audio_manifest_path": result["page_audio_manifest_path"],
                "page_timestamps_manifest_path": result["page_timestamps_manifest_path"],
                "page_video_manifest_path": result["page_video_manifest_path"],
                "final_video_path": result["final_video_path"],
            },
        )
        _report_gradio_progress(progress, 1.0, "Story video ready")
        return result
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(str(exc)) from exc
    finally:
        RUN_LOCK.release()


def _format_subtitle_status(page_video: dict) -> str:
    """Format one subtitle mode label for the parent inspector."""

    subtitle_mode = page_video["subtitle_mode"]
    fallback_reason = page_video.get("fallback_reason")
    if fallback_reason:
        return f"{subtitle_mode} ({fallback_reason})"
    return subtitle_mode


def _format_generation_outputs(result: dict) -> tuple:
    """Format pipeline results for both the child and parent views."""

    story = result["story_package"]
    page_images = result["page_images"]
    page_audio = result["page_audio"]
    page_timestamps = result["page_timestamps"]
    page_videos = result["page_videos"]
    synced_image_path = result["input_image"]

    return (
        synced_image_path,
        result["final_video_path"],
        result["working_image"],
        story.title,
        story.visual_canon,
        story.part_1.text,
        story.part_1.image_prompt,
        story.part_2.text,
        story.part_2.image_prompt,
        story.part_3.text,
        story.part_3.image_prompt,
        page_images[0]["image_path"],
        page_images[0]["final_prompt"],
        page_audio[0]["audio_path"],
        page_audio[0]["narration_text"],
        page_timestamps[0]["text"],
        page_timestamps[0]["words"],
        _format_subtitle_status(page_videos[0]),
        page_images[1]["image_path"],
        page_images[1]["final_prompt"],
        page_audio[1]["audio_path"],
        page_audio[1]["narration_text"],
        page_timestamps[1]["text"],
        page_timestamps[1]["words"],
        _format_subtitle_status(page_videos[1]),
        page_images[2]["image_path"],
        page_images[2]["final_prompt"],
        page_audio[2]["audio_path"],
        page_audio[2]["narration_text"],
        page_timestamps[2]["text"],
        page_timestamps[2]["words"],
        _format_subtitle_status(page_videos[2]),
        story.model_dump(mode="json"),
        result["final_video_path"],
    )


def _empty_generation_outputs(
    synced_image_path: str,
) -> tuple:
    """Clear the visible results before a new story run starts."""

    return (
        synced_image_path,
        None,
        None,
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        None,
        "",
        None,
        "",
        "",
        [],
        "",
        None,
        "",
        None,
        "",
        "",
        [],
        "",
        None,
        "",
        None,
        "",
        "",
        [],
        "",
        {},
        None,
    )


def generate_story_from_child(
    image_path: str,
    progress: gr.Progress = gr.Progress(),
):
    """Auto-run the child flow after an upload and sync the parent inspector."""

    yield _empty_generation_outputs(synced_image_path=image_path)
    yield _format_generation_outputs(
        _run_generation(image_path, progress=progress, source="child_mode")
    )


def generate_story_from_parent(
    image_path: str,
    progress: gr.Progress = gr.Progress(),
):
    """Run the parent/debug flow and sync the child-facing result view."""

    yield _empty_generation_outputs(synced_image_path=image_path)
    yield _format_generation_outputs(
        _run_generation(image_path, progress=progress, source="parent_mode")
    )


def build_demo() -> gr.Blocks:
    """Create the StoryAI Gradio interface with child and parent modes."""

    with gr.Blocks(title="StoryAI", css=APP_CSS) as demo:
        with gr.Column(elem_id="storyai-shell"):
            gr.HTML(
                """
                <div id="storyai-hero">
                  <h1>StoryAI</h1>
                  <p class="storyai-subtitle">Bedtime Story AI</p>
                  <p class="storyai-intro">
                    Start in Child Mode for a simple upload-and-go story video.
                    Switch to Parent Mode to inspect the working image, prompts,
                    audio, timestamps, and full artifact trail.
                  </p>
                </div>
                """
            )

            with gr.Tabs(elem_id="storyai-tabs"):
                with gr.Tab("Child Mode", render_children=True):
                    with gr.Column(elem_classes=["storyai-mode-panel"]):
                        gr.HTML(
                            """
                            <div class="storyai-tab-copy">
                              <p>
                                Upload or capture one drawing and StoryAI will start right away.
                                A progress bar will appear while the story video is being made.
                              </p>
                            </div>
                            """
                        )
                        child_image_input = gr.Image(
                            sources=["upload", "webcam"],
                            type="filepath",
                            label="Child drawing",
                            elem_classes=["storyai-media"],
                        )
                        child_final_video = gr.Video(
                            label="Story video",
                            elem_classes=["storyai-media"],
                        )

                with gr.Tab("Parent Mode", render_children=True):
                    with gr.Column(elem_classes=["storyai-mode-panel"]):
                        gr.HTML(
                            """
                            <div class="storyai-tab-copy">
                              <p>
                                Upload one drawing, then inspect the generated story,
                                prompts, audio, timestamps, and final video in one place.
                              </p>
                            </div>
                            """
                        )
                        with gr.Row(elem_classes=["storyai-media-row"], equal_height=True):
                            parent_image_input = gr.Image(
                                sources=["upload", "webcam"],
                                type="filepath",
                                label="Child drawing",
                                elem_classes=["storyai-media"],
                            )
                            working_image = gr.Image(
                                label="Normalized working image",
                                elem_classes=["storyai-media"],
                            )

                        with gr.Column(elem_classes=["storyai-controls"]):
                            generate_button = gr.Button(
                                "Generate Story + Final Video",
                                variant="primary",
                                elem_classes=["storyai-action"],
                            )

                        title_output = gr.Textbox(label="Title")
                        visual_canon_output = gr.Textbox(label="Visual canon", lines=3)

                        part_1_text = gr.Textbox(label="Part 1 text", lines=5)
                        part_1_prompt = gr.Textbox(label="Part 1 image prompt", lines=4)
                        part_2_text = gr.Textbox(label="Part 2 text", lines=5)
                        part_2_prompt = gr.Textbox(label="Part 2 image prompt", lines=4)
                        part_3_text = gr.Textbox(label="Part 3 text", lines=5)
                        part_3_prompt = gr.Textbox(label="Part 3 image prompt", lines=4)
                        story_json = gr.JSON(label="Story package JSON")

                        with gr.Tabs(elem_classes=["storyai-page-tabs"]):
                            with gr.Tab("Page 1"):
                                page_1_image = gr.Image(
                                    label="Page 1 image",
                                    elem_classes=["storyai-media"],
                                )
                                page_1_final_prompt = gr.Textbox(
                                    label="Page 1 final image generation prompt",
                                    lines=9,
                                )
                                page_1_audio = gr.Audio(label="Page 1 narration")
                                page_1_narration_text = gr.Textbox(
                                    label="Page 1 spoken narration text",
                                    lines=5,
                                )
                                page_1_transcript = gr.Textbox(
                                    label="Page 1 transcript",
                                    lines=5,
                                )
                                page_1_words = gr.JSON(label="Page 1 word timestamps")
                                page_1_subtitle_mode = gr.Textbox(label="Page 1 subtitle mode")

                            with gr.Tab("Page 2"):
                                page_2_image = gr.Image(
                                    label="Page 2 image",
                                    elem_classes=["storyai-media"],
                                )
                                page_2_final_prompt = gr.Textbox(
                                    label="Page 2 final image generation prompt",
                                    lines=9,
                                )
                                page_2_audio = gr.Audio(label="Page 2 narration")
                                page_2_narration_text = gr.Textbox(
                                    label="Page 2 spoken narration text",
                                    lines=5,
                                )
                                page_2_transcript = gr.Textbox(
                                    label="Page 2 transcript",
                                    lines=5,
                                )
                                page_2_words = gr.JSON(label="Page 2 word timestamps")
                                page_2_subtitle_mode = gr.Textbox(label="Page 2 subtitle mode")

                            with gr.Tab("Page 3"):
                                page_3_image = gr.Image(
                                    label="Page 3 image",
                                    elem_classes=["storyai-media"],
                                )
                                page_3_final_prompt = gr.Textbox(
                                    label="Page 3 final image generation prompt",
                                    lines=9,
                                )
                                page_3_audio = gr.Audio(label="Page 3 narration")
                                page_3_narration_text = gr.Textbox(
                                    label="Page 3 spoken narration text",
                                    lines=5,
                                )
                                page_3_transcript = gr.Textbox(
                                    label="Page 3 transcript",
                                    lines=5,
                                )
                                page_3_words = gr.JSON(label="Page 3 word timestamps")
                                page_3_subtitle_mode = gr.Textbox(label="Page 3 subtitle mode")

                        final_video = gr.Video(
                            label="Final story video",
                            elem_classes=["storyai-media"],
                        )

        shared_outputs = [
            child_final_video,
            working_image,
            title_output,
            visual_canon_output,
            part_1_text,
            part_1_prompt,
            part_2_text,
            part_2_prompt,
            part_3_text,
            part_3_prompt,
            page_1_image,
            page_1_final_prompt,
            page_1_audio,
            page_1_narration_text,
            page_1_transcript,
            page_1_words,
            page_1_subtitle_mode,
            page_2_image,
            page_2_final_prompt,
            page_2_audio,
            page_2_narration_text,
            page_2_transcript,
            page_2_words,
            page_2_subtitle_mode,
            page_3_image,
            page_3_final_prompt,
            page_3_audio,
            page_3_narration_text,
            page_3_transcript,
            page_3_words,
            page_3_subtitle_mode,
            story_json,
            final_video,
        ]

        child_image_input.upload(
            fn=generate_story_from_child,
            inputs=[child_image_input],
            outputs=[parent_image_input, *shared_outputs],
            api_name=False,
        )

        generate_button.click(
            fn=generate_story_from_parent,
            inputs=[parent_image_input],
            outputs=[child_image_input, *shared_outputs],
            api_name=False,
        )

    return demo


def launch_app() -> None:
    """Launch the app with sensible defaults for Colab and local runs."""

    model_config = get_runtime_model_config()
    demo = build_demo()
    share = os.getenv("STORYAI_SHARE")
    should_share = share.lower() == "true" if share else "COLAB_RELEASE_TAG" in os.environ

    print("Starting StoryAI...", flush=True)
    print(f"Share link enabled: {should_share}", flush=True)
    print(f"Artifacts directory: {DEFAULT_RUNS_DIR.resolve()}", flush=True)
    print(
        "Image mode: "
        f"{model_config['image_mode']} "
        f"({model_config['image_model']}, quality={model_config['image_quality']})",
        flush=True,
    )
    print(f"Story model: {model_config['story_model']}", flush=True)
    print(f"TTS model: {model_config['tts_model']}", flush=True)
    print(f"Timing model: {model_config['timing_model']}", flush=True)
    print(f"Terminal debug logging: {DEBUG_ENABLED}", flush=True)

    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="0.0.0.0",
        share=should_share,
        show_api=False,
    )


if __name__ == "__main__":
    launch_app()

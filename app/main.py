import os
import threading
import traceback

import gradio as gr

from .debug_utils import DEBUG_ENABLED, debug_print
from .pipeline import DEFAULT_RUNS_DIR, run_story_package_pipeline


RUN_LOCK = threading.Lock()


def generate_story(image_path: str):
    """Generate a structured story package from one uploaded image."""

    if not image_path:
        raise gr.Error("Upload or capture one drawing first.")

    if not RUN_LOCK.acquire(blocking=False):
        raise gr.Error("A run is already in progress. Please wait for it to finish.")

    try:
        debug_print(
            "GRADIO EVENT",
            {
                "input_image_path": image_path,
            },
        )
        result = run_story_package_pipeline(image_path)
        story = result["story_package"]
        page_images = result["page_images"]
        page_audio = result["page_audio"]
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
            },
        )

        return (
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
            page_images[1]["image_path"],
            page_images[1]["final_prompt"],
            page_audio[1]["audio_path"],
            page_images[2]["image_path"],
            page_images[2]["final_prompt"],
            page_audio[2]["audio_path"],
            story.model_dump(mode="json"),
            f"Artifacts saved to {result['run_dir']}",
        )
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(str(exc)) from exc
    finally:
        RUN_LOCK.release()


def build_demo() -> gr.Blocks:
    """Create the current StoryAI Gradio interface."""

    with gr.Blocks(title="StoryAI") as demo:
        gr.Markdown(
            """
            # StoryAI
            Upload or capture one child drawing, then generate a structured three-part bedtime story package and three page images.
            """
        )

        with gr.Row():
            image_input = gr.Image(
                sources=["upload", "webcam"],
                type="filepath",
                label="Child drawing",
            )
            working_image = gr.Image(label="Normalized working image")

        generate_button = gr.Button("Generate Story + Page Images", variant="primary")
        status_output = gr.Textbox(label="Run status")

        title_output = gr.Textbox(label="Title")
        visual_canon_output = gr.Textbox(label="Visual canon", lines=3)

        part_1_text = gr.Textbox(label="Part 1 text", lines=5)
        part_1_prompt = gr.Textbox(label="Part 1 image prompt", lines=4)
        part_2_text = gr.Textbox(label="Part 2 text", lines=5)
        part_2_prompt = gr.Textbox(label="Part 2 image prompt", lines=4)
        part_3_text = gr.Textbox(label="Part 3 text", lines=5)
        part_3_prompt = gr.Textbox(label="Part 3 image prompt", lines=4)
        story_json = gr.JSON(label="Story package JSON")

        with gr.Tabs():
            with gr.Tab("Page 1"):
                page_1_image = gr.Image(label="Page 1 image")
                page_1_final_prompt = gr.Textbox(
                    label="Page 1 final image generation prompt",
                    lines=9,
                )
                page_1_audio = gr.Audio(label="Page 1 narration")

            with gr.Tab("Page 2"):
                page_2_image = gr.Image(label="Page 2 image")
                page_2_final_prompt = gr.Textbox(
                    label="Page 2 final image generation prompt",
                    lines=9,
                )
                page_2_audio = gr.Audio(label="Page 2 narration")

            with gr.Tab("Page 3"):
                page_3_image = gr.Image(label="Page 3 image")
                page_3_final_prompt = gr.Textbox(
                    label="Page 3 final image generation prompt",
                    lines=9,
                )
                page_3_audio = gr.Audio(label="Page 3 narration")

        generate_button.click(
            fn=generate_story,
            inputs=[image_input],
            outputs=[
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
                page_2_image,
                page_2_final_prompt,
                page_2_audio,
                page_3_image,
                page_3_final_prompt,
                page_3_audio,
                story_json,
                status_output,
            ],
            api_name=False,
        )

    return demo


def launch_app() -> None:
    """Launch the app with sensible defaults for Colab and local runs."""

    demo = build_demo()
    share = os.getenv("STORYAI_SHARE")
    should_share = share.lower() == "true" if share else "COLAB_RELEASE_TAG" in os.environ

    print("Starting StoryAI...", flush=True)
    print(f"Share link enabled: {should_share}", flush=True)
    print(f"Artifacts directory: {DEFAULT_RUNS_DIR.resolve()}", flush=True)
    print(f"Terminal debug logging: {DEBUG_ENABLED}", flush=True)

    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="0.0.0.0",
        share=should_share,
        show_api=False,
    )


if __name__ == "__main__":
    launch_app()

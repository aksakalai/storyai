import os
import threading

import gradio as gr

from .pipeline import DEFAULT_RUNS_DIR, run_story_package_pipeline


RUN_LOCK = threading.Lock()


def generate_story(image_path: str):
    """Generate a structured story package from one uploaded image."""

    if not image_path:
        raise gr.Error("Upload or capture one drawing first.")

    if not RUN_LOCK.acquire(blocking=False):
        raise gr.Error("A run is already in progress. Please wait for it to finish.")

    try:
        result = run_story_package_pipeline(image_path)
        story = result["story_package"]

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
            story.model_dump(mode="json"),
            f"Artifacts saved to {result['run_dir']}",
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    finally:
        RUN_LOCK.release()


def build_demo() -> gr.Blocks:
    """Create the phase-one StoryAI Gradio interface."""

    with gr.Blocks(title="StoryAI") as demo:
        gr.Markdown(
            """
            # StoryAI
            Upload or capture one child drawing, then generate a structured three-part bedtime story package.
            """
        )

        with gr.Row():
            image_input = gr.Image(
                sources=["upload", "webcam"],
                type="filepath",
                label="Child drawing",
            )
            working_image = gr.Image(label="Normalized working image")

        generate_button = gr.Button("Generate Story Package", variant="primary")
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

    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="0.0.0.0",
        share=should_share,
        show_api=False,
    )


if __name__ == "__main__":
    launch_app()

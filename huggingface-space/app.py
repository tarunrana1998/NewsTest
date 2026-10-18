import os

import gradio as gr
from gradio_client import Client

MODEL_SPACE = os.environ.get("VIDEO_MODEL_SPACE", "Lightricks/ltx-video-distilled")


def generate_video(
    prompt: str,
    duration_seconds: int,
    guidance_scale: float,
) -> str:
    if not prompt.strip():
        raise gr.Error("Enter a scene prompt.")

    client = Client(MODEL_SPACE, hf_token=os.environ.get("HF_TOKEN") or None)
    result = client.predict(
        prompt,
        "worst quality, blurry, jittery, garbled text, logos, watermark",
        None,
        None,
        512,
        704,
        "text-to-video",
        duration_seconds,
        9,
        42,
        True,
        guidance_scale,
        True,
        api_name="/text_to_video",
    )
    video = result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else result[0]
    if isinstance(video, dict):
        video = video.get("video")
    if not isinstance(video, (str, os.PathLike)):
        raise gr.Error("The hosted video model returned no downloadable video.")
    return os.fspath(video)


with gr.Blocks() as demo:
    gr.Markdown("# News Video Generator")
    prompt_input = gr.Textbox(label="Scene prompt", lines=6)
    duration_input = gr.Slider(2, 7, value=7, step=1, label="Clip duration (seconds)")
    guidance_input = gr.Slider(1, 10, value=1, step=0.5, label="Guidance scale")
    generate_button = gr.Button("Generate video")
    video_output = gr.Video(label="Generated clip")

    generate_button.click(
        fn=generate_video,
        inputs=[prompt_input, duration_input, guidance_input],
        outputs=video_output,
        api_name="generate",
    )


demo.queue().launch()
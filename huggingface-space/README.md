---
title: News Video Generator
sdk: gradio
python_version: 3.10
app_file: app.py
pinned: false
---

# News Video Generator

This Space is a lightweight Gradio frontend that forwards prompts to the public
`Lightricks/ltx-video-distilled` Space. The video model runs on that remote
ZeroGPU Space; model weights are not downloaded to this Space or the local
NewsTest computer.

## Deploy

1. Create a new Gradio Space at https://huggingface.co/new-space.
2. Keep **CPU Basic** hardware selected. ZeroGPU hosting is not needed for this
   proxy.
3. Upload the files in this folder to the Space repository.
4. Wait for the Space to build, then set `HF_VIDEO_SPACE` in the NewsTest `.env`
   to your account and Space name.

Generation uses the upstream model Space's shared ZeroGPU quota, queue, and
availability. Its default output is a 2-second clip. A Hugging Face token may be
stored as the `HF_TOKEN` Space secret; do not commit it to the repository.
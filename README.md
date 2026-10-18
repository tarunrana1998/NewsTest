## LangChain chat

Add your Google AI Studio API key to `.env` and start the chat:

```sh
source .venv/bin/activate
python langchain_base.py
```

The default model is `gemini-2.5-flash`. Set `GEMINI_MODEL` in `.env` to select a
different Gemini model. Conversation history is kept for the duration of the
session; enter `exit` or `quit` to end it.

## Indian news video script agent

Generate a structured script from current headlines in The Hindu and The Indian
Express RSS feeds:

```sh
source .venv/bin/activate
python news_agent.py --duration 4 --language English --output news_script.json
```

Use `--topic` to focus the script, or omit `--output` to print JSON in the
terminal. Each story includes its publisher, publication time, source link,
key points, narration, and on-screen text. The feed summaries can be incomplete;
the agent is instructed not to invent missing facts.

## Talking Presenter Clips

`video_space.py` reads only each scene's `VOICEOVER`, creates Hindi speech with
the remote `hi-IN-MadhurNeural` voice, and lip-syncs it to the same synthetic
presenter portrait through the hosted
[Wav2Lip ZeroGPU Space](https://huggingface.co/spaces/pragnakalp/Wav2lip-ZeroGPU).
It does not send the visual b-roll prompts. Model weights stay remote; narration
audio is saved under `generated_videos/voiceovers/` and talking-head MP4s under
`generated_videos/talking_head/`.

The presenter portrait is `generated_videos/presenter_source.png`. Set these
entries in `.env` without replacing your Google API key:

```dotenv
HF_TALKING_HEAD_SPACE=pragnakalp/Wav2lip-ZeroGPU
HF_TOKEN=your-hugging-face-read-token
```

Keep the token secret. Install the small clients and create the first clip:

```sh
uv sync
uv run python video_space.py
```

Generate all seven scenes with `uv run python video_space.py --limit 7`. Resume
at a later scene with `--start-scene 3 --limit 5`. Hugging Face's free GPU
quota is limited; if a request is rejected, add a valid read token locally or
wait for the quota reset. Do not paste the token into chat.

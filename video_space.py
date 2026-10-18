import argparse
import asyncio
import os
import re
import shutil
from pathlib import Path
from typing import Any

import edge_tts
from dotenv import load_dotenv
from gradio_client import Client

load_dotenv()

SCENE_PATTERN = re.compile(
    r"^SCENE\s+(\d+)\s*\|\s*(.+?)\s*\n(.*?)(?=^SCENE\s+\d+\s*\||\Z)",
    re.MULTILINE | re.DOTALL,
)
VOICEOVER_PATTERN = re.compile(
    r"^VOICEOVER\s*\n(.*?)(?=^(?:ON-SCREEN TEXT|END CARD TEXT TO ADD IN EDITOR)\s*$|\Z)",
    re.MULTILINE | re.DOTALL,
)


def extract_scenes(prompt_file: Path) -> list[tuple[str, str]]:
    text = prompt_file.read_text(encoding="utf-8")
    scenes: list[tuple[str, str]] = []

    for match in SCENE_PATTERN.finditer(text):
        scene_number, title, scene_text = match.groups()
        voiceover_match = VOICEOVER_PATTERN.search(scene_text)
        if voiceover_match:
            voiceover = re.sub(r"\s+", " ", voiceover_match.group(1)).strip()
            safe_title = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()
            scenes.append((f"scene_{scene_number}_{safe_title}", voiceover))

    if not scenes:
        raise ValueError(f"No scene voiceovers found in {prompt_file}")
    return scenes


def copy_generated_files(value: Any, output_dir: Path, scene_name: str) -> list[Path]:
    copied: list[Path] = []
    seen: set[Path] = set()
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, (list, tuple)):
        children = value
    else:
        children = [value]

    for index, child in enumerate(children, start=1):
        if isinstance(child, (dict, list, tuple)):
            copied.extend(copy_generated_files(child, output_dir, scene_name))
            continue
        if not isinstance(child, (str, os.PathLike)):
            continue

        source = Path(child)
        if not source.is_file() or source.suffix.lower() not in {".mp4", ".webm", ".mov"}:
            continue
        resolved_source = source.resolve()
        if resolved_source in seen:
            continue
        seen.add(resolved_source)

        suffix = source.suffix or ".mp4"
        destination = output_dir / f"{scene_name}_{index}{suffix}"
        shutil.copy2(source, destination)
        copied.append(destination)

    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create presenter videos from the scene voiceovers using a hosted lip-sync Space."
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("news_video_scene_prompts.txt"),
        help="Scene prompt text file (default: news_video_scene_prompts.txt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated_videos"),
        help="Where presenter clips and narration audio are saved (default: generated_videos).",
    )
    parser.add_argument(
        "--portrait",
        type=Path,
        default=Path("generated_videos/presenter_source.png"),
        help="Synthetic presenter portrait used for every scene.",
    )
    parser.add_argument(
        "--show-api",
        action="store_true",
        help="Display the Space API and exit without generating clips.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Maximum number of scenes to generate (default: 1).",
    )
    parser.add_argument(
        "--start-scene",
        type=int,
        default=1,
        help="1-based scene number to start from (default: 1).",
    )
    parser.add_argument(
        "--voice",
        default="hi-IN-MadhurNeural",
        help="Remote Edge TTS voice (default: hi-IN-MadhurNeural).",
    )
    return parser.parse_args()


async def synthesize_voiceover(text: str, voice: str, output_path: Path) -> None:
    communicate = edge_tts.Communicate(text, voice, rate="-10%")
    await communicate.save(str(output_path))


def main() -> None:
    args = parse_args()
    space_id = os.environ.get("HF_TALKING_HEAD_SPACE", "pragnakalp/Wav2lip-ZeroGPU")

    client = Client(space_id, hf_token=os.environ.get("HF_TOKEN") or None)
    if args.show_api:
        client.view_api()
        return

    if args.limit < 1:
        raise SystemExit("--limit must be at least 1.")
    if args.start_scene < 1:
        raise SystemExit("--start-scene must be at least 1.")
    if not args.portrait.is_file():
        raise SystemExit(f"Presenter portrait not found: {args.portrait}")

    scenes = extract_scenes(args.prompts)
    start_index = args.start_scene - 1
    scenes = scenes[start_index : start_index + args.limit]
    if not scenes:
        raise SystemExit("No scenes selected; check --start-scene and --limit.")

    audio_dir = args.output_dir / "voiceovers"
    video_dir = args.output_dir / "talking_head"
    audio_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    for scene_name, voiceover in scenes:
        print(f"Generating presenter narration for {scene_name}...")
        try:
            audio_path = audio_dir / f"{scene_name}.mp3"
            asyncio.run(synthesize_voiceover(voiceover, args.voice, audio_path))
            result = client.predict(
                str(args.portrait),
                str(audio_path),
                api_name="/run_infrence",
            )
            videos = copy_generated_files(result, video_dir, scene_name)
            if not videos:
                raise RuntimeError(f"No talking-head video returned by Space: {result!r}")
            output_path = videos[0]
            print(f"Saved {output_path}")
        except Exception as error:
            raise SystemExit(f"Stopped at {scene_name}: {error}") from error


if __name__ == "__main__":
    main()
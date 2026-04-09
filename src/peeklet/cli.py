"""CLI entry point for Peeklet."""

from __future__ import annotations

from pathlib import Path

import click

import peeklet
from peeklet.config import apply_quality_preset, apply_sensitivity_preset, load_config
from peeklet.core.loader import load_frame
from peeklet.pipeline import Pipeline

VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm"}


def _write_context_outputs(
    source_name: str,
    duration: float,
    results: list,
    transcript_segments: list,
    output_dir: Path,
) -> None:
    """Write context.json and context.md to output_dir.

    Centralized so both image mode and video mode can produce the
    LLM-ready context artifacts. Imported lazily so the cost is paid
    only when actually called.
    """
    from peeklet.core.context_exporter import (
        build_context,
        write_context_json,
        write_context_markdown,
    )

    ctx = build_context(source_name, duration, results, transcript_segments)
    write_context_json(ctx, output_dir / "context.json")
    write_context_markdown(ctx, output_dir / "context.md")


def _detect_mode(input_path: Path, mode: str | None, image_extensions: set[str]) -> str:
    """Detect whether input is video or image mode."""
    if input_path.is_file():
        if input_path.suffix.lower() in VIDEO_EXTENSIONS:
            return "video"
        return "image"

    # Directory — scan contents
    has_videos = any(
        f.suffix.lower() in VIDEO_EXTENSIONS for f in input_path.iterdir() if f.is_file()
    )
    has_images = any(
        f.suffix.lower() in image_extensions for f in input_path.iterdir() if f.is_file()
    )

    if mode:
        return mode

    if has_videos and has_images:
        raise click.UsageError(
            "Directory contains both image and video files. "
            "Select a mode: --mode video or --mode image"
        )

    if has_videos:
        return "video"
    return "image"


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Video file, or directory containing images or videos.",
)
@click.option(
    "--output",
    "output_dir",
    type=click.Path(path_type=Path),
    default="./output",
    help="Directory for keyframes and manifest.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to config file (JSON or YAML).",
)
@click.option("--no-audio", is_flag=True, default=False, help="Disable audio detection for video.")
@click.option(
    "--mode",
    type=click.Choice(["video", "image"]),
    default=None,
    help="Force video or image mode (required for mixed directories).",
)
@click.option(
    "--transcript",
    "transcript_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to SRT or VTT transcript file.",
)
@click.option(
    "--demo-mode",
    "demo_mode",
    is_flag=True,
    default=False,
    help="Use the LLM to pick screenshot-worthy moments from the transcript "
    "and Peeklet to pick the actual frames. Requires --transcript and a video input.",
)
@click.option(
    "--llm-provider",
    "llm_provider",
    type=click.Choice(["anthropic", "openai", "openrouter"]),
    default=None,
    envvar="PEEKLET_LLM_PROVIDER",
    help="LLM provider for --demo-mode (anthropic, openai, or openrouter). "
    "OpenRouter requires OPENROUTER_API_KEY and uses namespaced model ids "
    "like 'anthropic/claude-3.5-sonnet'. "
    "Defaults to the value in config.demo_filter.llm_provider.",
)
@click.option(
    "--llm-model",
    "llm_model",
    type=str,
    default=None,
    envvar="PEEKLET_LLM_MODEL",
    help="LLM model identifier for --demo-mode. "
    "Defaults to the value in config.demo_filter.llm_model.",
)
@click.option(
    "--format",
    "keyframe_format",
    type=click.Choice(["png", "jpg"]),
    default=None,
    help="Keyframe image format (overrides config default).",
)
@click.option(
    "--quality",
    type=click.Choice(["fast", "balanced", "precise"]),
    default=None,
    help="Processing quality preset. Bundles processing_max_dim, "
    "sample_fps, and frame_search_resolution.",
)
@click.option(
    "--sensitivity",
    type=click.Choice(["low", "medium", "high"]),
    default=None,
    help="Change-detection sensitivity preset. Bundles ssim_threshold, "
    "min_changed_pct, and min_changed_blocks.",
)
@click.version_option(version=peeklet.__version__, prog_name="peeklet")
def main(
    input_path: Path,
    output_dir: Path,
    config_path: Path | None,
    demo_mode: bool,
    keyframe_format: str | None,
    llm_model: str | None,
    llm_provider: str | None,
    mode: str | None,
    no_audio: bool,
    quality: str | None,
    sensitivity: str | None,
    transcript_path: Path | None,
) -> None:
    """Smart screenshot change detection.

    Filters noise from screenshot sequences or video recordings
    and exports a structured Parquet manifest of keyframes.
    """
    config = load_config(config_path)
    if no_audio:
        config.video.audio_detection = False
    if transcript_path:
        config.video.transcript_path = str(transcript_path)
    config.exporter.output_dir = str(output_dir)
    if keyframe_format is not None:
        config.exporter.keyframe_format = keyframe_format  # type: ignore[assignment]

    if quality is not None:
        apply_quality_preset(config, quality)

    if sensitivity is not None:
        apply_sensitivity_preset(config, sensitivity)

    if demo_mode:
        if not transcript_path:
            raise click.UsageError(
                "--demo-mode requires --transcript. Demo mode needs both a "
                "video and a transcript to filter frames effectively."
            )
        config.demo_filter.enabled = True
        if llm_provider is not None:
            config.demo_filter.llm_provider = llm_provider  # type: ignore[assignment]
        if llm_model is not None:
            config.demo_filter.llm_model = llm_model

    image_extensions = {f".{fmt}" for fmt in config.input.supported_formats}
    detected_mode = _detect_mode(input_path, mode, image_extensions)

    if demo_mode and detected_mode != "video":
        raise click.UsageError("--demo-mode only applies to video inputs.")

    if detected_mode == "video":
        _run_video_mode(input_path, config)
    else:
        _run_image_mode(input_path, config, image_extensions)


def _run_video_mode(input_path: Path, config: peeklet.config.PeekletConfig) -> None:
    """Process video file(s)."""
    from peeklet.core.audio import parse_transcript
    from peeklet.core.exporter import ManifestWriter
    from peeklet.core.transcript_trigger import detect_triggers
    from peeklet.core.video import process_video

    # Detect transcript triggers if transcript is provided.
    # Skip in demo mode — the demo pipeline picks moments from the LLM,
    # not from keyword heuristics, and ignores forced_timestamps anyway.
    forced_timestamps: list[float] = []
    if config.video.transcript_path and not config.demo_filter.enabled:
        segments = parse_transcript(Path(config.video.transcript_path))
        triggers = detect_triggers(segments)
        forced_timestamps = [t.timestamp for t in triggers]
        if triggers:
            click.echo(f"Found {len(triggers)} transcript trigger(s)")

    if input_path.is_file():
        video_files = [input_path]
    else:
        video_files = sorted(
            f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        )

    if not video_files:
        click.echo(f"No video files found in {input_path}")
        return

    if len(video_files) > 1 and config.video.transcript_path:
        raise click.UsageError(
            "--transcript can only be used with a single video file, "
            f"not a directory of {len(video_files)} videos."
        )

    click.echo(f"Processing {len(video_files)} video(s)")

    output_dir = Path(config.exporter.output_dir)
    writer = ManifestWriter(
        path=output_dir / "manifest.parquet",
        compression=config.exporter.parquet_compression,
    )

    total_keyframes = 0
    for vf in video_files:
        click.echo(f"  Processing: {vf.name}")
        results = process_video(vf, config, writer=writer, forced_timestamps=forced_timestamps)
        kf_count = sum(1 for r in results if r.is_keyframe)
        total_keyframes += kf_count
        click.echo(f"    {kf_count} keyframes extracted")

    writer.flush()

    click.echo(
        f"Done: {total_keyframes} total keyframes. Manifest: {output_dir / 'manifest.parquet'}"
    )
    click.echo(f"Context: {output_dir / 'context.json'}, {output_dir / 'context.md'}")


def _run_image_mode(
    input_dir: Path,
    config: peeklet.config.PeekletConfig,
    extensions: set[str],
) -> None:
    """Process image directory (existing behavior)."""
    pipeline = Pipeline(config)
    files = sorted(f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in extensions)

    if not files:
        click.echo(f"No supported images found in {input_dir}")
        return

    click.echo(f"Processing {len(files)} frames from {input_dir}")

    keyframe_count = 0
    results: list = []
    for f in files:
        frame = load_frame(f)
        result = pipeline.process_frame(frame, frame_id=f.stem, source_format=f.suffix.lstrip("."))
        results.append(result)
        if result.is_keyframe:
            keyframe_count += 1

    pipeline.finalize()
    output_dir = Path(config.exporter.output_dir)

    # Image mode has no inherent timeline. We pass duration=0.0 as a
    # sentinel (image input is a sequence, not a stream) and use the
    # input directory name as the "source" identifier so downstream
    # context consumers have something to display. Transcript segments
    # are always empty for image input — there's no narration to align.
    _write_context_outputs(
        source_name=input_dir.name,
        duration=0.0,
        results=results,
        transcript_segments=[],
        output_dir=output_dir,
    )

    click.echo(
        f"Done: {keyframe_count} keyframes from {len(files)} frames "
        f"({100 * keyframe_count / len(files):.1f}%). "
        f"Manifest: {output_dir / 'manifest.parquet'}"
    )
    click.echo(f"Context: {output_dir / 'context.json'}, {output_dir / 'context.md'}")

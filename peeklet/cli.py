"""CLI entry point for Peeklet."""

from __future__ import annotations

from pathlib import Path

import click

import peeklet
from peeklet.config import load_config
from peeklet.core.loader import load_frame
from peeklet.pipeline import Pipeline

VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm"}


def _detect_mode(
    input_path: Path, mode: str | None, image_extensions: set[str]
) -> str:
    """Detect whether input is video or image mode."""
    if input_path.is_file():
        if input_path.suffix.lower() in VIDEO_EXTENSIONS:
            return "video"
        return "image"

    # Directory — scan contents
    has_videos = any(
        f.suffix.lower() in VIDEO_EXTENSIONS
        for f in input_path.iterdir() if f.is_file()
    )
    has_images = any(
        f.suffix.lower() in image_extensions
        for f in input_path.iterdir() if f.is_file()
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
@click.option("--no-redact", is_flag=True, default=False, help="Disable PII redaction.")
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
@click.version_option(version=peeklet.__version__, prog_name="peeklet")
def main(
    input_path: Path,
    output_dir: Path,
    config_path: Path | None,
    no_redact: bool,
    no_audio: bool,
    mode: str | None,
    transcript_path: Path | None,
) -> None:
    """Smart screenshot change detection.

    Filters noise from screenshot sequences or video recordings,
    redacts PII, and exports a structured Parquet manifest of keyframes.
    """
    config = load_config(config_path)
    if no_redact:
        config.redactor.enabled = False
    if no_audio:
        config.video.audio_detection = False
    if transcript_path:
        config.video.transcript_path = str(transcript_path)
    config.exporter.output_dir = str(output_dir)

    image_extensions = {f".{fmt}" for fmt in config.input.supported_formats}
    detected_mode = _detect_mode(input_path, mode, image_extensions)

    if detected_mode == "video":
        _run_video_mode(input_path, config)
    else:
        _run_image_mode(input_path, config, image_extensions)


def _run_video_mode(input_path: Path, config: peeklet.config.PeekletConfig) -> None:
    """Process video file(s)."""
    from peeklet.core.exporter import ManifestWriter
    from peeklet.core.video import process_video

    if input_path.is_file():
        video_files = [input_path]
    else:
        video_files = sorted(
            f for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        )

    if not video_files:
        click.echo(f"No video files found in {input_path}")
        return

    click.echo(f"Processing {len(video_files)} video(s)")

    output_dir = Path(config.exporter.output_dir)
    writer = ManifestWriter(
        path=output_dir / "manifest.parquet",
        compression=config.exporter.parquet_compression,
    )

    total_keyframes = 0
    for vf in video_files:
        click.echo(f"  Processing: {vf.name}")
        results = process_video(vf, config, writer=writer)
        kf_count = sum(1 for r in results if r.is_keyframe)
        total_keyframes += kf_count
        click.echo(f"    {kf_count} keyframes extracted")

    writer.flush()

    output_dir = Path(config.exporter.output_dir)
    click.echo(
        f"Done: {total_keyframes} total keyframes. "
        f"Manifest: {output_dir / 'manifest.parquet'}"
    )


def _run_image_mode(
    input_dir: Path,
    config: peeklet.config.PeekletConfig,
    extensions: set[str],
) -> None:
    """Process image directory (existing behavior)."""
    pipeline = Pipeline(config)
    files = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    )

    if not files:
        click.echo(f"No supported images found in {input_dir}")
        return

    click.echo(f"Processing {len(files)} frames from {input_dir}")

    keyframe_count = 0
    for f in files:
        frame = load_frame(f)
        result = pipeline.process_frame(
            frame, frame_id=f.stem, source_format=f.suffix.lstrip(".")
        )
        if result.is_keyframe:
            keyframe_count += 1

    pipeline.finalize()
    output_dir = Path(config.exporter.output_dir)
    click.echo(
        f"Done: {keyframe_count} keyframes from {len(files)} frames "
        f"({100 * keyframe_count / len(files):.1f}%). "
        f"Manifest: {output_dir / 'manifest.parquet'}"
    )

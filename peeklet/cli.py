"""CLI entry point for Peeklet."""

from __future__ import annotations

from pathlib import Path

import click

import peeklet
from peeklet.config import load_config
from peeklet.core.loader import load_frame
from peeklet.pipeline import Pipeline


@click.command()
@click.option(
    "--input",
    "input_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing screenshot images.",
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
@click.version_option(version=peeklet.__version__, prog_name="peeklet")
def main(input_dir: Path, output_dir: Path, config_path: Path | None, no_redact: bool) -> None:
    """Smart screenshot change detection.

    Filters noise from screenshot sequences, redacts PII, and exports
    a structured Parquet manifest of keyframes.
    """
    config = load_config(config_path)
    if no_redact:
        config.redactor.enabled = False
    config.exporter.output_dir = str(output_dir)

    pipeline = Pipeline(config)

    extensions = {f".{fmt}" for fmt in config.input.supported_formats}
    files = sorted(f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in extensions)

    if not files:
        click.echo(f"No supported images found in {input_dir}")
        return

    click.echo(f"Processing {len(files)} frames from {input_dir}")

    keyframe_count = 0
    for f in files:
        frame = load_frame(f)
        result = pipeline.process_frame(frame, frame_id=f.stem, source_format=f.suffix.lstrip("."))
        if result.is_keyframe:
            keyframe_count += 1

    pipeline.finalize()
    click.echo(
        f"Done: {keyframe_count} keyframes from {len(files)} frames "
        f"({100 * keyframe_count / len(files):.1f}%). "
        f"Manifest: {output_dir / 'manifest.parquet'}"
    )

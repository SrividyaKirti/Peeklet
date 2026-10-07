"""peeklet: annotate a screen-recording transcript with screenshots."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from peeklet.annotate import annotate
from peeklet.config import load_config
from peeklet.screen import MissingDependencyError
from peeklet.transcript import TranscriptError


@click.command()
@click.version_option(package_name="peeklet")
@click.argument("video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--transcript",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="SRT, VTT or Fathom markdown transcript.",
)
@click.option(
    "--out",
    "out_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Output directory for transcript.json and frame images.",
)
@click.option(
    "--max-images",
    type=click.IntRange(min=0),
    default=None,
    help="Maximum screenshots (default 20).",
)
@click.option("--no-llm", is_flag=True, help="Heuristic selection only; no descriptions.")
@click.option(
    "--llm-provider",
    type=click.Choice(["anthropic", "openai", "openrouter"]),
    default=None,
    help="LLM provider for screen judging and descriptions (default anthropic).",
)
@click.option("--llm-model", default=None, help="Model id (default claude-haiku-4-5).")
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a Peeklet config file (YAML or JSON) overriding the defaults.",
)
@click.option("--debug", is_flag=True, help="Also write debug.json with scoring details.")
def main(
    video: Path,
    transcript: Path,
    out_dir: Path,
    max_images: int | None,
    no_llm: bool,
    llm_provider: str | None,
    llm_model: str | None,
    config_path: Path | None,
    debug: bool,
) -> None:
    """Annotate VIDEO's transcript with the screenshots it refers to."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    cfg = load_config(config_path)
    if llm_provider:
        cfg.llm_provider = llm_provider  # type: ignore[assignment]
    if llm_model:
        cfg.llm_model = llm_model
    try:
        entries = annotate(
            video,
            transcript,
            out_dir,
            max_images=max_images,
            config=cfg,
            use_llm=not no_llm,
            debug=debug,
        )
    except (TranscriptError, MissingDependencyError) as exc:
        raise click.ClickException(str(exc)) from exc
    s = entries.stats
    n_warn = len(s.warnings)
    click.echo(
        f"{s.lines} lines, {s.screens_found} screens found, {s.shortlisted} shortlisted, "
        f"{s.kept} kept, {n_warn} warning{'' if n_warn == 1 else 's'} "
        f"-> {out_dir / 'transcript.json'}"
    )

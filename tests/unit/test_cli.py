"""CLI smoke tests (annotate is mocked)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

from click.testing import CliRunner

from peeklet.cli import main
from peeklet.render import Entries, RunStats
from peeklet.transcript import TranscriptError

if TYPE_CHECKING:
    from pathlib import Path


def _files(tmp: Path) -> tuple[Path, Path]:
    v, t = tmp / "v.mp4", tmp / "t.vtt"
    v.write_bytes(b"x")
    t.write_text("WEBVTT\n")
    return v, t


def test_cli_passes_options_and_prints_summary(tmp_path: Path) -> None:
    v, t = _files(tmp_path)
    fake = Entries([], tmp_path / "out", RunStats(12, 5, 4, 3, ["w"]))
    with mock.patch("peeklet.cli.annotate", return_value=fake) as ann:
        res = CliRunner().invoke(
            main,
            [
                str(v),
                "--transcript",
                str(t),
                "--out",
                str(tmp_path / "out"),
                "--max-images",
                "7",
                "--no-llm",
                "--llm-model",
                "claude-sonnet-5-5",
                "--debug",
            ],
        )
    assert res.exit_code == 0, res.output
    kwargs = ann.call_args.kwargs
    assert kwargs["max_images"] == 7 and kwargs["use_llm"] is False and kwargs["debug"] is True
    assert kwargs["config"].llm_model == "claude-sonnet-5-5"
    assert "12 lines, 5 screens found, 4 shortlisted, 3 kept, 1 warning" in res.output


def test_cli_reports_transcript_error(tmp_path: Path) -> None:
    v, t = _files(tmp_path)
    with mock.patch("peeklet.cli.annotate", side_effect=TranscriptError("bad transcript")):
        res = CliRunner().invoke(main, [str(v), "--transcript", str(t), "--out", str(tmp_path)])
    assert res.exit_code != 0
    assert "bad transcript" in res.output


def test_cli_requires_transcript(tmp_path: Path) -> None:
    v, _ = _files(tmp_path)
    res = CliRunner().invoke(main, [str(v), "--out", str(tmp_path)])
    assert res.exit_code != 0

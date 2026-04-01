"""Tests for the CLI interface."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from click.testing import CliRunner
from PIL import Image

from peeklet.cli import main


def _create_test_images(directory: Path, count: int = 5) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        frame = np.full((100, 100, 3), i * 50, dtype=np.uint8)
        Image.fromarray(frame).save(directory / f"frame_{i:03d}.png")


class TestCli:
    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Smart screenshot change detection" in result.output

    def test_batch_mode(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _create_test_images(input_dir, count=5)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--input", str(input_dir), "--output", str(output_dir), "--no-redact"],
        )
        assert result.exit_code == 0

        manifest = output_dir / "manifest.parquet"
        assert manifest.exists()
        table = pq.read_table(manifest)
        assert table.num_rows == 5

    def test_batch_with_config(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _create_test_images(input_dir, count=3)

        config_file = tmp_path / "config.json"
        config_file.write_text(
            '{"comparator": {"ssim_threshold": 0.5}, "redactor": {"enabled": false}}'
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--input", str(input_dir), "--output", str(output_dir), "--config", str(config_file)],
        )
        assert result.exit_code == 0

    def test_missing_input_dir_errors(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--input", "/nonexistent/dir"])
        assert result.exit_code != 0

    def test_version(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

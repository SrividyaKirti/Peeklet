"""Tests for the CLI interface."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner
from PIL import Image

from peeklet.cli import main

try:
    import av  # noqa: F401

    _has_video_deps = True
except ImportError:
    _has_video_deps = False


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
            ["--input", str(input_dir), "--output", str(output_dir)],
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
        config_file.write_text('{"comparator": {"ssim_threshold": 0.5}}')

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--input", str(input_dir), "--output", str(output_dir), "--config", str(config_file)],
        )
        assert result.exit_code == 0

    def test_image_mode_writes_context_files(self, tmp_path: Path) -> None:
        """Image mode produces context.json and context.md alongside the
        manifest, with real content (not just empty placeholder files)."""
        import json

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _create_test_images(input_dir, count=5)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--input", str(input_dir), "--output", str(output_dir)],
        )
        assert result.exit_code == 0, result.output

        assert (output_dir / "manifest.parquet").exists()
        assert (output_dir / "context.json").exists()
        assert (output_dir / "context.md").exists()

        # Verify the context.json has real content, not an empty placeholder.
        ctx = json.loads((output_dir / "context.json").read_text())
        assert "video" in ctx
        assert "screenshots" in ctx
        # The 5 progressively-different test frames should yield at least one
        # keyframe / screenshot. If results.append in _run_image_mode regresses,
        # this assertion catches it.
        assert ctx["video"]["total_screenshots"] >= 1
        assert len(ctx["screenshots"]) >= 1
        assert ctx["video"]["filename"] == "input"

        # The context.md should be non-empty.
        md_content = (output_dir / "context.md").read_text()
        assert len(md_content) > 0

    def test_format_flag_png_produces_png_keyframes(self, tmp_path: Path) -> None:
        """--format png produces .png keyframe images."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _create_test_images(input_dir, count=5)

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
                "--format",
                "png",
            ],
        )
        assert result.exit_code == 0, result.output

        png_files = list(output_dir.glob("*.png"))
        jpg_files = list(output_dir.glob("*.jpg"))
        assert len(png_files) >= 1, "expected at least one PNG keyframe"
        assert len(jpg_files) == 0, "no JPG keyframes should be present"

    def test_format_flag_jpg_produces_jpg_keyframes(self, tmp_path: Path) -> None:
        """--format jpg (the existing default) produces .jpg keyframe images."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _create_test_images(input_dir, count=5)

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
                "--format",
                "jpg",
            ],
        )
        assert result.exit_code == 0, result.output

        jpg_files = list(output_dir.glob("*.jpg"))
        png_files = list(output_dir.glob("*.png"))
        assert len(jpg_files) >= 1, "expected at least one JPG keyframe"
        assert len(png_files) == 0

    def test_missing_input_dir_errors(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--input", "/nonexistent/dir"])
        assert result.exit_code != 0

    def test_version(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


@pytest.mark.skipif(not _has_video_deps, reason="requires peeklet[video]")
class TestVideoCliDetection:
    def test_video_file_input(self, tmp_path: Path) -> None:
        """CLI accepts a video file as --input."""
        import imageio.v3 as iio

        video_path = tmp_path / "demo.mp4"
        frames = [np.zeros((60, 80, 3), dtype=np.uint8)] * 10
        with iio.imopen(video_path, "w", plugin="pyav") as out:
            out.init_video_stream("libx264", fps=10)
            for f in frames:
                out.write_frame(f)

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--input",
                str(video_path),
                "--output",
                str(tmp_path / "output"),
            ],
        )
        assert result.exit_code == 0
        assert "keyframe" in result.output.lower()

    def test_directory_with_only_videos(self, tmp_path: Path) -> None:
        """CLI processes directory of video files."""
        import imageio.v3 as iio

        for name in ["a.mp4", "b.mp4"]:
            video_path = tmp_path / name
            frames = [np.zeros((60, 80, 3), dtype=np.uint8)] * 10
            with iio.imopen(video_path, "w", plugin="pyav") as out:
                out.init_video_stream("libx264", fps=10)
                for f in frames:
                    out.write_frame(f)

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--input",
                str(tmp_path),
                "--output",
                str(tmp_path / "output"),
            ],
        )
        assert result.exit_code == 0

    def test_mixed_directory_without_mode_errors(self, tmp_path: Path) -> None:
        """CLI errors on mixed directory without --mode."""
        import imageio.v3 as iio

        video_path = tmp_path / "demo.mp4"
        frames = [np.zeros((60, 80, 3), dtype=np.uint8)] * 10
        with iio.imopen(video_path, "w", plugin="pyav") as out:
            out.init_video_stream("libx264", fps=10)
            for f in frames:
                out.write_frame(f)

        img = Image.new("RGB", (80, 60))
        img.save(tmp_path / "shot.png")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--input",
                str(tmp_path),
                "--output",
                str(tmp_path / "output"),
            ],
        )
        assert result.exit_code != 0
        assert "--mode" in result.output

    def test_mixed_directory_with_mode_video(self, tmp_path: Path) -> None:
        """CLI processes only videos when --mode video is specified."""
        import imageio.v3 as iio

        video_path = tmp_path / "demo.mp4"
        frames = [np.zeros((60, 80, 3), dtype=np.uint8)] * 10
        with iio.imopen(video_path, "w", plugin="pyav") as out:
            out.init_video_stream("libx264", fps=10)
            for f in frames:
                out.write_frame(f)

        img = Image.new("RGB", (80, 60))
        img.save(tmp_path / "shot.png")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--input",
                str(tmp_path),
                "--output",
                str(tmp_path / "output"),
                "--mode",
                "video",
            ],
        )
        assert result.exit_code == 0

    def test_transcript_flag(self, tmp_path: Path) -> None:
        """CLI accepts --transcript flag."""
        import imageio.v3 as iio

        video_path = tmp_path / "demo.mp4"
        frames = [np.zeros((60, 80, 3), dtype=np.uint8)] * 10
        with iio.imopen(video_path, "w", plugin="pyav") as out:
            out.init_video_stream("libx264", fps=10)
            for f in frames:
                out.write_frame(f)

        srt_path = tmp_path / "transcript.srt"
        srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n\n")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--input",
                str(video_path),
                "--output",
                str(tmp_path / "output"),
                "--transcript",
                str(srt_path),
            ],
        )
        assert result.exit_code == 0

    def test_video_with_transcript_full_cli_flow(self, tmp_path: Path) -> None:
        """CLI processes video + transcript and produces context files."""
        import imageio.v3 as iio

        video_path = tmp_path / "demo.mp4"
        # Two solid color frames - guarantees one visual change keyframe
        frames = [np.zeros((60, 80, 3), dtype=np.uint8)] * 30 + [
            np.full((60, 80, 3), 255, dtype=np.uint8)
        ] * 30
        with iio.imopen(video_path, "w", plugin="pyav") as out:
            out.init_video_stream("libx264", fps=30)
            for f in frames:
                out.write_frame(f)

        srt_path = tmp_path / "transcript.srt"
        srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nLook at this dashboard here\n\n")

        output_dir = tmp_path / "output"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--input",
                str(video_path),
                "--output",
                str(output_dir),
                "--transcript",
                str(srt_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "transcript trigger" not in result.output.lower()
        assert (output_dir / "context.json").exists()
        assert (output_dir / "context.md").exists()

    def test_directory_with_transcript_errors(self, tmp_path: Path) -> None:
        """CLI errors when --transcript is used with a directory of videos."""
        import imageio.v3 as iio

        for name in ["a.mp4", "b.mp4"]:
            video_path = tmp_path / name
            frames = [np.zeros((60, 80, 3), dtype=np.uint8)] * 10
            with iio.imopen(video_path, "w", plugin="pyav") as out:
                out.init_video_stream("libx264", fps=10)
                for f in frames:
                    out.write_frame(f)

        srt_path = tmp_path / "transcript.srt"
        srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n\n")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--input",
                str(tmp_path),
                "--output",
                str(tmp_path / "output"),
                "--transcript",
                str(srt_path),
            ],
        )
        assert result.exit_code != 0
        assert "transcript" in result.output.lower()
        assert "single video" in result.output.lower()


@pytest.mark.skipif(not _has_video_deps, reason="requires peeklet[video]")
def test_cli_demo_mode_requires_transcript(tmp_path):
    """--demo-mode without --transcript exits with a clear error."""
    from click.testing import CliRunner

    from peeklet.cli import main
    from tests.unit.helpers_video import write_synthetic_video

    video_path = tmp_path / "v.mp4"
    write_synthetic_video(video_path, duration_sec=2, fps=10, width=64, height=64)

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["--input", str(video_path), "--output", str(tmp_path / "out"), "--demo-mode"],
    )
    assert result.exit_code == 2
    assert "--demo-mode requires --transcript" in result.output


def test_cli_demo_mode_requires_video(tmp_path):
    """--demo-mode without a video input exits with a clear error."""
    from click.testing import CliRunner

    from peeklet.cli import main

    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    (images_dir / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    transcript = tmp_path / "t.srt"
    transcript.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--input",
            str(images_dir),
            "--output",
            str(tmp_path / "out"),
            "--mode",
            "image",
            "--transcript",
            str(transcript),
            "--demo-mode",
        ],
    )
    assert result.exit_code == 2
    assert "--demo-mode only applies to video inputs" in result.output


@pytest.mark.skipif(not _has_video_deps, reason="requires peeklet[video]")
def test_cli_demo_mode_propagates_provider_and_model(tmp_path, monkeypatch):
    """--llm-provider and --llm-model end up on config.demo_filter."""
    from click.testing import CliRunner

    from peeklet.cli import main
    from peeklet.core import video as video_module
    from tests.unit.helpers_video import write_synthetic_video

    video_path = tmp_path / "v.mp4"
    write_synthetic_video(video_path, duration_sec=2, fps=10, width=64, height=64)
    transcript = tmp_path / "t.srt"
    transcript.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n")

    captured = {}

    real_process_video = video_module.process_video

    def spy_process_video(path, config, **kwargs):
        captured["enabled"] = config.demo_filter.enabled
        captured["provider"] = config.demo_filter.llm_provider
        captured["model"] = config.demo_filter.llm_model
        return real_process_video(path, config, **kwargs)

    monkeypatch.setattr(video_module, "process_video", spy_process_video)
    # Stub out the demo filter so the test doesn't need a real LLM call.
    # Patch on video_module — that's where process_video has imported the
    # symbol. Patching df_module would not affect the already-imported reference.
    # Return (screens, moments) tuple matching the new apply_demo_filter signature.
    monkeypatch.setattr(video_module, "apply_demo_filter", lambda **_kw: ([], []))

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--input",
            str(video_path),
            "--output",
            str(tmp_path / "out"),
            "--no-audio",
            "--transcript",
            str(transcript),
            "--demo-mode",
            "--llm-provider",
            "openai",
            "--llm-model",
            "gpt-4o-mini",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured == {
        "enabled": True,
        "provider": "openai",
        "model": "gpt-4o-mini",
    }


@pytest.mark.skipif(not _has_video_deps, reason="requires peeklet[video]")
def test_cli_demo_mode_accepts_openrouter_provider(tmp_path, monkeypatch):
    """--llm-provider openrouter is accepted by Click and propagates to config."""
    from click.testing import CliRunner

    from peeklet.cli import main
    from peeklet.core import video as video_module
    from tests.unit.helpers_video import write_synthetic_video

    video_path = tmp_path / "v.mp4"
    write_synthetic_video(video_path, duration_sec=2, fps=10, width=64, height=64)
    transcript = tmp_path / "t.srt"
    transcript.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n")

    captured = {}

    real_process_video = video_module.process_video

    def spy_process_video(path, config, **kwargs):
        captured["provider"] = config.demo_filter.llm_provider
        captured["model"] = config.demo_filter.llm_model
        return real_process_video(path, config, **kwargs)

    monkeypatch.setattr(video_module, "process_video", spy_process_video)
    # Return (screens, moments) tuple matching the new apply_demo_filter signature.
    monkeypatch.setattr(video_module, "apply_demo_filter", lambda **_kw: ([], []))

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--input",
            str(video_path),
            "--output",
            str(tmp_path / "out"),
            "--no-audio",
            "--transcript",
            str(transcript),
            "--demo-mode",
            "--llm-provider",
            "openrouter",
            "--llm-model",
            "anthropic/claude-3.5-sonnet",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured == {
        "provider": "openrouter",
        "model": "anthropic/claude-3.5-sonnet",
    }


@pytest.mark.skipif(not _has_video_deps, reason="requires peeklet[video]")
def test_non_demo_transcript_is_silent_no_op(tmp_path, monkeypatch):
    """In non-demo mode, --transcript no longer drives screenshot selection.

    The CLI reads the file (config wiring) but invokes no keyword detector
    and produces the same output it would without --transcript.
    """
    from click.testing import CliRunner

    from peeklet.cli import main
    from tests.unit.helpers_video import write_synthetic_video

    video_path = tmp_path / "v.mp4"
    write_synthetic_video(video_path, duration_sec=2, fps=10, width=64, height=64)
    srt_path = tmp_path / "t.srt"
    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nLook at this dashboard here\n")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--input",
            str(video_path),
            "--output",
            str(tmp_path / "out"),
            "--no-audio",
            "--transcript",
            str(srt_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "transcript trigger" not in result.output.lower()


@pytest.mark.skipif(not _has_video_deps, reason="requires peeklet[video]")
def test_cli_quality_preset_fast_applies_values(tmp_path, monkeypatch):
    """--quality fast sets processing_max_dim and sample_fps."""
    from click.testing import CliRunner

    from peeklet.cli import main
    from peeklet.core import video as video_module
    from tests.unit.helpers_video import write_synthetic_video

    video_path = tmp_path / "v.mp4"
    write_synthetic_video(video_path, duration_sec=2, fps=10, width=64, height=64)

    captured = {}
    real_process_video = video_module.process_video

    def spy_process_video(path, config, **kwargs):
        captured["max_dim"] = config.video.processing_max_dim
        captured["sample_fps"] = config.video.sample_fps
        return real_process_video(path, config, **kwargs)

    monkeypatch.setattr(video_module, "process_video", spy_process_video)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--input",
            str(video_path),
            "--output",
            str(tmp_path / "out"),
            "--no-audio",
            "--quality",
            "fast",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured == {"max_dim": 480, "sample_fps": 0.5}


@pytest.mark.skipif(not _has_video_deps, reason="requires peeklet[video]")
def test_cli_no_quality_flag_uses_current_defaults(tmp_path, monkeypatch):
    """No --quality flag = today's defaults (which by design match the
    'balanced' preset). This is a regression guard against accidental
    drift in the underlying VideoConfig defaults — it does NOT exercise
    apply_quality_preset (the if-gate skips it when --quality is absent)."""
    from click.testing import CliRunner

    from peeklet.cli import main
    from peeklet.core import video as video_module
    from tests.unit.helpers_video import write_synthetic_video

    video_path = tmp_path / "v.mp4"
    write_synthetic_video(video_path, duration_sec=2, fps=10, width=64, height=64)

    captured = {}
    real_process_video = video_module.process_video

    def spy_process_video(path, config, **kwargs):
        captured["max_dim"] = config.video.processing_max_dim
        captured["sample_fps"] = config.video.sample_fps
        return real_process_video(path, config, **kwargs)

    monkeypatch.setattr(video_module, "process_video", spy_process_video)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--input",
            str(video_path),
            "--output",
            str(tmp_path / "out"),
            "--no-audio",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured == {"max_dim": 720, "sample_fps": 1.0}


@pytest.mark.skipif(not _has_video_deps, reason="requires peeklet[video]")
def test_cli_sensitivity_preset_high_applies_values(tmp_path, monkeypatch):
    """--sensitivity high sets the comparator thresholds correctly."""
    from click.testing import CliRunner

    from peeklet.cli import main
    from peeklet.core import video as video_module
    from tests.unit.helpers_video import write_synthetic_video

    video_path = tmp_path / "v.mp4"
    write_synthetic_video(video_path, duration_sec=2, fps=10, width=64, height=64)

    captured = {}
    real_process_video = video_module.process_video

    def spy_process_video(path, config, **kwargs):
        captured["ssim"] = config.comparator.ssim_threshold
        captured["pct"] = config.comparator.min_changed_pct
        captured["blocks"] = config.comparator.min_changed_blocks
        return real_process_video(path, config, **kwargs)

    monkeypatch.setattr(video_module, "process_video", spy_process_video)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--input",
            str(video_path),
            "--output",
            str(tmp_path / "out"),
            "--no-audio",
            "--sensitivity",
            "high",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured == {"ssim": 0.75, "pct": 1.0, "blocks": 2}

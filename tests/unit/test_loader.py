"""Tests for the image loader module."""

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from peeklet.core.loader import load_frame


def _save_pil(img: Image.Image, path: Path, fmt: str) -> Path:
    img.save(path, format=fmt)
    return path


def _make_test_image() -> Image.Image:
    return Image.fromarray(np.arange(80 * 60 * 3, dtype=np.uint8).reshape(60, 80, 3))


class TestLoadFromPath:
    def test_load_png(self, tmp_path: Path) -> None:
        img = _make_test_image()
        path = _save_pil(img, tmp_path / "test.png", "PNG")
        result = load_frame(path)
        assert result.shape == (60, 80, 3)
        assert result.dtype == np.uint8

    def test_load_jpeg(self, tmp_path: Path) -> None:
        img = _make_test_image()
        path = _save_pil(img, tmp_path / "test.jpg", "JPEG")
        result = load_frame(path)
        assert result.shape == (60, 80, 3)

    def test_load_bmp(self, tmp_path: Path) -> None:
        img = _make_test_image()
        path = _save_pil(img, tmp_path / "test.bmp", "BMP")
        result = load_frame(path)
        assert result.shape == (60, 80, 3)

    def test_load_webp(self, tmp_path: Path) -> None:
        img = _make_test_image()
        path = _save_pil(img, tmp_path / "test.webp", "WEBP")
        result = load_frame(path)
        assert result.shape == (60, 80, 3)

    def test_load_tiff(self, tmp_path: Path) -> None:
        img = _make_test_image()
        path = _save_pil(img, tmp_path / "test.tiff", "TIFF")
        result = load_frame(path)
        assert result.shape == (60, 80, 3)

    def test_load_string_path(self, tmp_path: Path) -> None:
        img = _make_test_image()
        path = _save_pil(img, tmp_path / "test.png", "PNG")
        result = load_frame(str(path))
        assert result.shape == (60, 80, 3)

    def test_load_nonexistent_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_frame("/nonexistent/image.png")

    def test_load_corrupt_file_raises(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "corrupt.png"
        corrupt.write_bytes(b"not an image")
        with pytest.raises(ValueError, match="Could not load"):
            load_frame(corrupt)


class TestLoadFromMemory:
    def test_load_numpy_rgb(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = load_frame(frame)
        assert result.shape == (100, 100, 3)
        np.testing.assert_array_equal(result, frame)

    def test_load_numpy_rgba(self) -> None:
        frame = np.zeros((100, 100, 4), dtype=np.uint8)
        result = load_frame(frame)
        assert result.shape == (100, 100, 3)

    def test_load_numpy_grayscale(self) -> None:
        frame = np.zeros((100, 100), dtype=np.uint8)
        result = load_frame(frame)
        assert result.shape == (100, 100, 3)

    def test_load_pil_image(self) -> None:
        img = Image.new("RGB", (80, 60), color=(128, 64, 32))
        result = load_frame(img)
        assert result.shape == (60, 80, 3)
        assert result[0, 0, 0] == 128

    def test_load_pil_rgba(self) -> None:
        img = Image.new("RGBA", (80, 60), color=(128, 64, 32, 255))
        result = load_frame(img)
        assert result.shape == (60, 80, 3)

    def test_load_bytes(self, tmp_path: Path) -> None:
        img = _make_test_image()
        buf = BytesIO()
        img.save(buf, format="PNG")
        result = load_frame(buf.getvalue())
        assert result.shape == (60, 80, 3)

    def test_load_invalid_bytes_raises(self) -> None:
        with pytest.raises(ValueError, match="Could not load"):
            load_frame(b"not an image")

    def test_load_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Unsupported source type"):
            load_frame(12345)

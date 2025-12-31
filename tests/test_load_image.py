### This file is used to test the load_image function in the read_write.py file
### This can be tested by pytest.

import numpy as np
import torch
import cv2
import pytest
from PIL import Image

from imtools import load_image


# -----------------------------
# Helpers
# -----------------------------

def save_png(path, array):
    cv2.imwrite(str(path), array)


# -----------------------------
# File-based inputs (OpenCV)
# -----------------------------

def test_load_rgb_image_file_raw(tmp_path):
    img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    path = tmp_path / "rgb.png"
    save_png(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    out = load_image(str(path))

    # OpenCV default: BGR, uint8, HWC
    assert out.shape == (32, 32, 3)
    assert out.dtype == np.uint8


def test_load_rgb_image_file_convert_rgb(tmp_path):
    img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    path = tmp_path / "rgb.png"
    save_png(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    out = load_image(
        str(path),
        convert_to_rgb=True,
        normalize_dims=True,
    )

    assert out.shape == (32, 32, 3)
    assert out.dtype == np.uint8


def test_load_rgba_file_preserve_alpha(tmp_path):
    img = np.random.randint(0, 255, (16, 16, 4), dtype=np.uint8)
    path = tmp_path / "rgba.png"
    save_png(path, cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))

    out = load_image(str(path))

    assert out.shape == (16, 16, 4)


def test_load_rgba_file_drop_alpha(tmp_path):
    img = np.random.randint(0, 255, (16, 16, 4), dtype=np.uint8)
    path = tmp_path / "rgba.png"
    save_png(path, cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))

    out = load_image(
        str(path),
        drop_alpha=True,
        normalize_dims=True,
    )

    assert out.shape == (16, 16, 3)


def test_load_grayscale_file_no_normalization(tmp_path):
    img = np.random.randint(0, 255, (20, 20), dtype=np.uint8)
    path = tmp_path / "gray.png"
    save_png(path, img)

    out = load_image(str(path))

    assert out.shape == (20, 20)


def test_load_grayscale_file_to_rgb(tmp_path):
    img = np.random.randint(0, 255, (20, 20), dtype=np.uint8)
    path = tmp_path / "gray.png"
    save_png(path, img)

    out = load_image(
        str(path),
        normalize_dims=True,
        convert_to_rgb=True,
    )

    assert out.shape == (20, 20, 3)


# -----------------------------
# PIL inputs
# -----------------------------

def test_pil_rgb_raw():
    img = Image.fromarray(
        np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8),
        mode="RGB",
    )

    out = load_image(img)

    assert isinstance(out, np.ndarray)
    assert out.shape == (10, 10, 3)


def test_pil_rgba_raw():
    img = Image.fromarray(
        np.random.randint(0, 255, (10, 10, 4), dtype=np.uint8),
        mode="RGBA",
    )

    out = load_image(img)

    assert out.shape == (10, 10, 4)


def test_pil_rgba_convert_rgb():
    img = Image.fromarray(
        np.random.randint(0, 255, (10, 10, 4), dtype=np.uint8),
        mode="RGBA",
    )

    out = load_image(
        img,
        convert_to_rgb=True,
        normalize_dims=True,
    )

    assert out.shape == (10, 10, 3)


# -----------------------------
# NumPy inputs
# -----------------------------

def test_numpy_grayscale_raw():
    img = np.random.randint(0, 255, (12, 12), dtype=np.uint8)

    out = load_image(img)

    assert out.shape == (12, 12)


def test_numpy_grayscale_to_rgb():
    img = np.random.randint(0, 255, (12, 12), dtype=np.uint8)

    out = load_image(
        img,
        normalize_dims=True,
        convert_to_rgb=True,
    )

    assert out.shape == (12, 12, 3)


def test_numpy_rgba_preserve_alpha():
    img = np.random.randint(0, 255, (12, 12, 4), dtype=np.uint8)

    out = load_image(img)

    assert out.shape == (12, 12, 4)


# -----------------------------
# Torch inputs
# -----------------------------

def test_torch_chw_raw():
    img = torch.randint(0, 255, (3, 24, 24), dtype=torch.uint8)

    out = load_image(img)

    assert out.shape == (3, 24, 24)


def test_torch_chw_to_hwc_rgb():
    img = torch.randint(0, 255, (3, 24, 24), dtype=torch.uint8)

    out = load_image(
        img,
        normalize_dims=True,
        convert_to_rgb=True,
    )

    assert out.shape == (24, 24, 3)


def test_torch_float_preserve_dtype():
    img = torch.rand(3, 16, 16)

    out = load_image(img)

    assert out.dtype == np.float32


def test_torch_float_to_uint8():
    img = torch.rand(3, 16, 16)

    out = load_image(
        img,
        normalize_dims=True,
        convert_to_rgb=True,
        force_uint8=True,
    )

    assert out.dtype == np.uint8
    assert out.max() <= 255
    assert out.min() >= 0


# -----------------------------
# Dtype handling
# -----------------------------

def test_bool_image_no_force_uint8():
    img = np.zeros((6, 6), dtype=bool)
    img[2:4, 2:4] = True

    out = load_image(img)

    assert out.dtype == np.bool_


def test_bool_image_force_uint8():
    img = np.zeros((6, 6), dtype=bool)
    img[2:4, 2:4] = True

    out = load_image(
        img,
        force_uint8=True,
    )

    assert out.dtype == np.uint8
    assert set(np.unique(out)) <= {0, 255}


# -----------------------------
# Errors
# -----------------------------

def test_invalid_input_type():
    with pytest.raises(TypeError):
        load_image(12345)

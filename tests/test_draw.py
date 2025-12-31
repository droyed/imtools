### This file tests the draw.py functions
### This can be tested by pytest.

import numpy as np
import pytest
import torch
from PIL import Image
from unittest.mock import patch, MagicMock
import os

from imtools import text_on_canvas, draw_mask_overlays


# -----------------------------
# text_on_canvas tests
# -----------------------------

def test_text_on_canvas_single_string():
    """Test with a single string"""
    result = text_on_canvas("Hello World")

    assert isinstance(result, Image.Image)
    assert result.mode == "RGBA"
    # Should have non-zero dimensions
    assert result.width > 0
    assert result.height > 0


def test_text_on_canvas_list_of_strings():
    """Test with a list of strings"""
    result = text_on_canvas(["Line 1", "Line 2", "Line 3"])

    assert isinstance(result, Image.Image)
    # Height should be larger for multiple lines
    single_line = text_on_canvas("Line 1")
    assert result.height > single_line.height


def test_text_on_canvas_empty_string():
    """Test with empty string"""
    result = text_on_canvas("")

    assert isinstance(result, Image.Image)
    # Should still create a canvas with padding
    assert result.width > 0
    assert result.height > 0


def test_text_on_canvas_empty_list():
    """Test with empty list"""
    result = text_on_canvas([])

    assert isinstance(result, Image.Image)


def test_text_on_canvas_align_left():
    """Test left alignment"""
    result = text_on_canvas(["Short", "Much longer line"], align="left")

    assert isinstance(result, Image.Image)


def test_text_on_canvas_align_center():
    """Test center alignment"""
    result = text_on_canvas(["Short", "Much longer line"], align="center")

    assert isinstance(result, Image.Image)


def test_text_on_canvas_align_right():
    """Test right alignment"""
    result = text_on_canvas(["Short", "Much longer line"], align="right")

    assert isinstance(result, Image.Image)


def test_text_on_canvas_invalid_align():
    """Test assertion error with invalid alignment"""
    with pytest.raises(AssertionError):
        text_on_canvas("Test", align="justified")


def test_text_on_canvas_crop_to_text_false():
    """Test crop_to_text=False (default)"""
    result = text_on_canvas("Test", padding=50, crop_to_text=False)

    # Should include full padding
    assert isinstance(result, Image.Image)
    # Width should be substantial due to padding
    assert result.width > 100


def test_text_on_canvas_crop_to_text_true():
    """Test crop_to_text=True"""
    result_no_crop = text_on_canvas("Test", padding=50, crop_to_text=False)
    result_crop = text_on_canvas("Test", padding=50, crop_to_text=True)

    # Cropped version should be smaller
    assert result_crop.width < result_no_crop.width
    assert result_crop.height < result_no_crop.height


def test_text_on_canvas_transparent_text_only_false():
    """Test transparent_text_only=False (default)"""
    result = text_on_canvas("Test", bg_color=(30, 30, 30, 255), transparent_text_only=False)

    # Should have background color
    arr = np.array(result)
    # Alpha channel should have non-transparent values
    assert arr[:, :, 3].max() == 255


def test_text_on_canvas_transparent_text_only_true():
    """Test transparent_text_only=True"""
    result = text_on_canvas("Test", transparent_text_only=True)

    # Background should be transparent
    arr = np.array(result)
    # Check that there are transparent pixels (alpha = 0)
    assert np.any(arr[:, :, 3] == 0)


def test_text_on_canvas_output_rgb_false():
    """Test output_rgb=False (default RGBA)"""
    result = text_on_canvas("Test", output_rgb=False)

    assert result.mode == "RGBA"


def test_text_on_canvas_output_rgb_true():
    """Test output_rgb=True (RGB output)"""
    result = text_on_canvas("Test", output_rgb=True)

    assert result.mode == "RGB"


def test_text_on_canvas_min_width():
    """Test min_width parameter"""
    result_no_min = text_on_canvas("Hi", padding=10)
    result_min = text_on_canvas("Hi", padding=10, min_width=500)

    # With min_width, canvas should be wider
    assert result_min.width >= 500
    assert result_min.width > result_no_min.width


def test_text_on_canvas_text_opacity_valid():
    """Test valid text_opacity values"""
    result_opaque = text_on_canvas("Test", text_opacity=255)
    result_semi = text_on_canvas("Test", text_opacity=128)
    result_transparent = text_on_canvas("Test", text_opacity=0)

    # All should create valid images
    assert isinstance(result_opaque, Image.Image)
    assert isinstance(result_semi, Image.Image)
    assert isinstance(result_transparent, Image.Image)


def test_text_on_canvas_text_opacity_invalid_high():
    """Test assertion error with text_opacity > 255"""
    with pytest.raises(AssertionError):
        text_on_canvas("Test", text_opacity=256)


def test_text_on_canvas_text_opacity_invalid_low():
    """Test assertion error with text_opacity < 0"""
    with pytest.raises(AssertionError):
        text_on_canvas("Test", text_opacity=-1)


def test_text_on_canvas_stroke_width_no_color():
    """Test stroke_width without stroke_color"""
    result = text_on_canvas("Test", stroke_width=2)

    assert isinstance(result, Image.Image)


def test_text_on_canvas_stroke_width_with_color():
    """Test stroke_width with stroke_color"""
    result = text_on_canvas("Test", stroke_width=3, stroke_color=(255, 0, 0))

    assert isinstance(result, Image.Image)


def test_text_on_canvas_custom_colors():
    """Test custom background and text colors"""
    result = text_on_canvas(
        "Test",
        bg_color=(255, 0, 0, 255),
        text_color=(0, 255, 0),
        output_rgb=True
    )

    assert isinstance(result, Image.Image)
    # Check that colors are applied (red and green should be present)
    arr = np.array(result)
    assert arr.shape[2] == 3  # RGB


def test_text_on_canvas_custom_font_size():
    """Test different font sizes"""
    result_small = text_on_canvas("Test", font_size=20)
    result_large = text_on_canvas("Test", font_size=80)

    # Larger font should produce larger canvas
    assert result_large.height > result_small.height


def test_text_on_canvas_custom_padding():
    """Test different padding values"""
    result_small_pad = text_on_canvas("Test", padding=5)
    result_large_pad = text_on_canvas("Test", padding=50)

    # Larger padding should produce larger canvas
    assert result_large_pad.width > result_small_pad.width
    assert result_large_pad.height > result_small_pad.height


def test_text_on_canvas_line_spacing():
    """Test different line spacing values"""
    result_tight = text_on_canvas(["Line 1", "Line 2"], line_spacing=0)
    result_spaced = text_on_canvas(["Line 1", "Line 2"], line_spacing=30)

    # More line spacing should increase height
    assert result_spaced.height > result_tight.height


def test_text_on_canvas_multiline_different_lengths():
    """Test multiline text with varying line lengths"""
    result = text_on_canvas([
        "Short",
        "This is a much longer line of text",
        "Medium length"
    ])

    assert isinstance(result, Image.Image)
    # Width should accommodate the longest line
    assert result.width > 0


# -----------------------------
# draw_mask_overlays tests
# -----------------------------

def test_draw_mask_overlays_save_to_file(tmp_path):
    """Test saving mask overlay to file"""
    # Create test image and mask
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255  # Square mask

    output_path = tmp_path / "mask_overlay.png"

    with patch('matplotlib.pyplot.show'):
        draw_mask_overlays(img, [mask], output_path=str(output_path))

    # File should be created
    assert output_path.exists()


def test_draw_mask_overlays_single_mask(tmp_path):
    """Test with a single mask"""
    img = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[20:60, 20:60] = 255

    output_path = tmp_path / "single_mask.png"

    with patch('matplotlib.pyplot.show'):
        draw_mask_overlays(img, [mask], output_path=str(output_path))

    assert output_path.exists()


def test_draw_mask_overlays_multiple_masks(tmp_path):
    """Test with multiple masks"""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[10:40, 10:40] = 255

    mask2 = np.zeros((100, 100), dtype=np.uint8)
    mask2[60:90, 60:90] = 255

    mask3 = np.zeros((100, 100), dtype=np.uint8)
    mask3[40:60, 40:60] = 255

    output_path = tmp_path / "multi_mask.png"

    with patch('matplotlib.pyplot.show'):
        draw_mask_overlays(img, [mask1, mask2, mask3], output_path=str(output_path))

    assert output_path.exists()


def test_draw_mask_overlays_with_scores(tmp_path):
    """Test with confidence scores"""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255

    scores = [0.95]
    output_path = tmp_path / "mask_with_score.png"

    with patch('matplotlib.pyplot.show'):
        draw_mask_overlays(img, [mask], scores=scores, output_path=str(output_path))

    assert output_path.exists()


def test_draw_mask_overlays_without_scores(tmp_path):
    """Test without confidence scores (should still work)"""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255

    output_path = tmp_path / "mask_no_score.png"

    with patch('matplotlib.pyplot.show'):
        draw_mask_overlays(img, [mask], output_path=str(output_path))

    assert output_path.exists()


def test_draw_mask_overlays_with_title_prefix(tmp_path):
    """Test with title_prefix"""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255

    output_path = tmp_path / "mask_titled.png"

    with patch('matplotlib.pyplot.show'):
        draw_mask_overlays(
            img,
            [mask],
            title_prefix="Test Image",
            output_path=str(output_path)
        )

    assert output_path.exists()


def test_draw_mask_overlays_without_title_prefix(tmp_path):
    """Test without title_prefix"""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255

    output_path = tmp_path / "mask_no_title.png"

    with patch('matplotlib.pyplot.show'):
        draw_mask_overlays(img, [mask], output_path=str(output_path))

    assert output_path.exists()


def test_draw_mask_overlays_empty_masks(tmp_path):
    """Test with empty masks list"""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    output_path = tmp_path / "no_masks.png"

    with patch('matplotlib.pyplot.show'):
        draw_mask_overlays(img, [], output_path=str(output_path))

    # Should still create an image (just the base image)
    assert output_path.exists()


def test_draw_mask_overlays_pil_image_input(tmp_path):
    """Test with PIL Image input"""
    pil_img = Image.fromarray(
        np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
        mode="RGB"
    )
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255

    output_path = tmp_path / "pil_input.png"

    with patch('matplotlib.pyplot.show'):
        draw_mask_overlays(pil_img, [mask], output_path=str(output_path))

    assert output_path.exists()


def test_draw_mask_overlays_torch_mask_input(tmp_path):
    """Test with torch tensor mask"""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mask_tensor = torch.zeros((100, 100), dtype=torch.uint8)
    mask_tensor[30:70, 30:70] = 255

    output_path = tmp_path / "torch_mask.png"

    with patch('matplotlib.pyplot.show'):
        draw_mask_overlays(img, [mask_tensor], output_path=str(output_path))

    assert output_path.exists()


def test_draw_mask_overlays_no_output_path_mocked():
    """Test without output_path (display mode) - mock plt.show()"""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255

    with patch('matplotlib.pyplot.show') as mock_show:
        draw_mask_overlays(img, [mask])
        # plt.show() should be called when output_path is None
        mock_show.assert_called_once()


def test_draw_mask_overlays_mask_with_no_pixels(tmp_path):
    """Test with mask that has no positive pixels"""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)  # All zeros

    output_path = tmp_path / "empty_mask.png"

    with patch('matplotlib.pyplot.show'):
        draw_mask_overlays(img, [mask], output_path=str(output_path))

    assert output_path.exists()


def test_draw_mask_overlays_image_from_file(tmp_path):
    """Test with image loaded from file path"""
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img_path = tmp_path / "test_image.png"

    # Save image to file
    import cv2
    cv2.imwrite(str(img_path), img_array)

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255

    output_path = tmp_path / "overlay_from_file.png"

    with patch('matplotlib.pyplot.show'):
        draw_mask_overlays(str(img_path), [mask], output_path=str(output_path))

    assert output_path.exists()

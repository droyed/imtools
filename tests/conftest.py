"""
Shared pytest fixtures for imtools tests.
"""
import numpy as np
import pytest
from pathlib import Path
from PIL import Image


@pytest.fixture
def sample_image():
    """Returns a small RGB image (64x64x3 uint8)."""
    return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_grayscale_image():
    """Returns a small grayscale image (64x64 uint8)."""
    return np.random.randint(0, 256, (64, 64), dtype=np.uint8)


@pytest.fixture
def sample_binary_mask():
    """Returns a small binary mask (64x64)."""
    mask = np.zeros((64, 64), dtype=np.bool_)
    mask[10:30, 10:30] = True
    return mask


@pytest.fixture
def sample_multiclass_mask():
    """Returns a small multi-class mask (64x64 with values 0-3)."""
    mask = np.zeros((64, 64), dtype=np.int32)
    mask[0:16, 0:16] = 1
    mask[16:32, 16:32] = 2
    mask[32:48, 32:48] = 3
    return mask


@pytest.fixture
def sample_bbox_list():
    """Returns a list of bounding boxes as dicts."""
    return [
        {'x': 10, 'y': 20, 'width': 50, 'height': 30, 'label': 'person', 'score': 0.95},
        {'x': 100, 'y': 150, 'width': 80, 'height': 120, 'label': 'car', 'score': 0.87},
    ]


@pytest.fixture
def sample_masks_stack():
    """Returns a stack of boolean masks (3, 64, 64)."""
    masks = np.zeros((3, 64, 64), dtype=np.bool_)
    masks[0, 10:20, 10:20] = True
    masks[1, 30:40, 30:40] = True
    masks[2, 50:60, 50:60] = True
    return masks


@pytest.fixture
def tmp_image_file(tmp_path, sample_image):
    """Writes sample_image to a temp PNG and yields the path."""
    image_path = tmp_path / "test_image.png"
    Image.fromarray(sample_image).save(image_path)
    yield image_path


@pytest.fixture
def tmp_yolo_dir(tmp_path):
    """Creates a temp directory with YOLO format label files and dummy images."""
    yolo_dir = tmp_path / "yolo_dataset"
    yolo_dir.mkdir()

    # Create label directory
    labels_dir = yolo_dir / "labels"
    labels_dir.mkdir()

    # Create image directory
    images_dir = yolo_dir / "images"
    images_dir.mkdir()

    # Create sample YOLO label files
    # Format: class_id x_center y_center width height (normalized)
    label_content_1 = "0 0.5 0.5 0.3 0.4\n1 0.2 0.3 0.1 0.15"
    label_content_2 = "0 0.7 0.8 0.2 0.1"

    (labels_dir / "image1.txt").write_text(label_content_1)
    (labels_dir / "image2.txt").write_text(label_content_2)

    # Create dummy images (10x10 grayscale)
    dummy_img = np.zeros((10, 10), dtype=np.uint8)
    Image.fromarray(dummy_img).save(images_dir / "image1.png")
    Image.fromarray(dummy_img).save(images_dir / "image2.png")

    return yolo_dir

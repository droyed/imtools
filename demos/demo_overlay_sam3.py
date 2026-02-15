#!/usr/bin/env python
# coding: utf-8
"""
Demo: Overlay SAM3 Visualization

This demo shows how to use imtools overlay features to visualize SAM3 segmentation
results with various customization options.

Usage:
    python -m demos.demo_overlay_sam3
"""

import os

# =============================================================================
# Section 1: Imports and Setup
# =============================================================================

from imtools import (
    LabelFormat, BlendConfig, TitleConfig, LabelStyle,
    overlay_visualize, yolo_to_annotations, yolo_to_label_image,
)
from setup_demo_data import get_pedestrian_sample_sam
from demo_config import get_output_dir

# Ensure outputs directory exists
output_dir = get_output_dir('overlay_sam3')
os.makedirs(output_dir, exist_ok=True)

# Load sample data
img, results = get_pedestrian_sample_sam()

# Convert SAM3 results to label image
label_image = yolo_to_label_image(results)


# =============================================================================
# Section 2: Basic Usage
# =============================================================================

overlay_visualize(
    img,
    label_image,
    title="Minimal Setup",
    savepath=os.path.join(output_dir, 'Sam3_Minimal_Setup.png')
)


# =============================================================================
# Section 3: Adding Annotations
# =============================================================================

annotations = yolo_to_annotations(results, label_format=LabelFormat.MULTI_COORDS)

overlay_visualize(
    img,
    label_image,
    annotations,
    title="Annotations",
    savepath=os.path.join(output_dir, 'Sam3_Annotations.png')
)


# =============================================================================
# Section 4: Using Presets
# =============================================================================

my_labelstyle_config = LabelStyle.Presets.high_contrast()
my_blend_config = BlendConfig.Presets.bold()
my_title_config = TitleConfig.Presets.high_contrast()

overlay_visualize(
    img=img,
    label_image=label_image,
    annotations=annotations,
    blend_config=my_blend_config,
    label_style=my_labelstyle_config,
    title_config=my_title_config,
    title="Preset Setup",
    savepath=os.path.join(output_dir, 'Sam3_Preset_Setup.png')
)


# =============================================================================
# Section 5: Advanced Customization
# =============================================================================

# A. Label Style Configuration
my_labelstyle_config = LabelStyle(
    font_size=12,
    padding=2,
    alpha=255,
    show_boxes=True,
    text_color='white',
    box_fill=(120, 120, 120),
    box_outline='red',
    box_outline_width=1,
    font_name='times.ttf'
)

# B. Blend Settings
my_blend_config = BlendConfig.from_params(
    alpha=0.92,
    method='colormap',
    colormap='hsv'
)

# C. Title Settings
my_title_config = TitleConfig(
    font_size=16,
    text_color=(255, 255, 255),
    bg_color=(0, 0, 0),        # Pitch black title bar
    align='center'             # Center alignment
)

# Create overlay with fully customized settings
overlay_visualize(
    img=img,
    label_image=label_image,
    annotations=annotations,
    blend_config=my_blend_config,
    label_style=my_labelstyle_config,
    title_config=my_title_config,
    title="Customized Setup",
    savepath=os.path.join(output_dir, 'Sam3_Customized_Setup.png')
)

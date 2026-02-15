# ## Overlay Usage

# ## Get started
# Run the cells below to step through the imtools overlay features.

# ### 1. Imports and Data Loading

import os
from imtools.label_formats import LabelFormat
from imtools.common import BlendConfig, TitleConfig, LabelStyle
from imtools import overlay_visualize
from setup_demo_data import get_pedestrian_sample_yolo
from imtools.annotations import yolo_to_annotations
from imtools.converters import yolo_to_label_image
from demo_config import get_output_dir

# Ensure outputs directory exists
output_dir = get_output_dir('overlay_yolo')
os.makedirs(output_dir, exist_ok=True)


img, results = get_pedestrian_sample_yolo()


label_image = yolo_to_label_image(results)


# ### 2.1 Create an overlay image with the minimal setup of image and label image
overlay_visualize(img, label_image, title="Minimal Setup", savepath=os.path.join(output_dir, 'Yolo_Minimal_Setup.png'))

# ### 2.2 Add annotations
annotations = yolo_to_annotations(results, label_format=LabelFormat.MULTI_COORDS)

overlay_visualize(img, label_image, annotations, title="Annotations", savepath=os.path.join(output_dir, 'Yolo_Annotations.png'))

# 2.3 Customize configurations for editing blending, labeling and title styles with Presets
# Presets
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
    savepath=os.path.join(output_dir, 'Yolo_Preset_Setup.png')
)

# ### 2.4 Advanced Customization
# Manually configure every aspect of the label style, blending settings, and title bar.

# ### 2.4.1 Customize configurations with direct parameters for each config

# A. Label Style
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
    align='center'               # center alignment
)

# Create the overlay image
overlay_visualize(
    img=img, 
    label_image=label_image, 
    annotations=annotations,
    blend_config=my_blend_config,
    label_style=my_labelstyle_config,
    title_config=my_title_config,
    title="Customized Setup",
    savepath=os.path.join(output_dir, 'Yolo_Customized_Setup.png')
)


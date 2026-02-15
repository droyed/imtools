#!/usr/bin/env python
# coding: utf-8
import os

# ## Overlay Usage

# ## Get started
# Run the cells below to step through the imtools overlay features.

# ### 1. Imports and Data Loading

# In[1]:


from imtools.converters import binary_mask_to_label_image
from imtools.annotations import label_image_to_annotations
from imtools.label_formats import LabelFormat
from imtools.common import BlendConfig, TitleConfig, LabelStyle
from imtools import overlay_visualize
from setup_demo_data import get_coins_sample
from demo_config import get_output_dir

# Ensure outputs directory exists
output_dir = get_output_dir('overlay_mask')
os.makedirs(output_dir, exist_ok=True)

img, mask = get_coins_sample()


# ### 2. Minimal Setup
# Create a label image from a mask and generate a basic overlay.

# In[2]:


# 1. Setup label image from mask/YOLO/SAM results
label_image = binary_mask_to_label_image(mask)

# 2. Create an overlay image with the minimal setup of image and label image
overlay_visualize(img, label_image, title="Minimal Setup", savepath=os.path.join(output_dir, 'Mask_Minimal_Setup.png'))


# ### 3. Adding Annotations

# In[3]:


# 2.1 Add annotations
annotations = label_image_to_annotations(
    label_image, class_name='', label_format=LabelFormat.CLASS_HASHINDEX
)
overlay_visualize(img, label_image, annotations, title="Annotations", savepath=os.path.join(output_dir, 'Mask_Annotations.png'))


# ### 4. Using Presets
# Customize configurations for blending, labeling, and title styles using built-in presets.

# In[4]:


# 2.2 Customize configurations for editing blending, labeling and title styles with Presets
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
    savepath=os.path.join(output_dir, 'Mask_Preset_Setup.png')
)


# ### 5. Advanced Customization
# Manually configure every aspect of the label style, blending settings, and title bar.

# In[5]:


# 2.3 Customize configurations with direct parameters for each config

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
    savepath=os.path.join(output_dir, 'Mask_Customized_Setup.png')
)


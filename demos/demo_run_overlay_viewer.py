import numpy as np
from skimage import data
from scipy.ndimage import binary_closing
from imtools.mask_utils import generate_label_image

# Setup image and mask
threshold=150
kernel_size=7
img = data.coins()
mask = img > threshold
img = np.repeat(img[:,:,None],3, axis=-1)

kernel = np.ones((kernel_size, kernel_size))
mask = binary_closing(mask, structure=kernel)

# Create label image
labelimg = generate_label_image(mask, connectivity=8)    

# Display overlay on viewer
import imtools
blended_img, settings = imtools.viz.run_overlay_viewer(img, labelimg, initial_method='colormap', initial_category='Qualitative', initial_colormap='tab20', initial_alpha=0.3)

import cv2
import numpy as np
from .read_write import load_image


def stack_images(image_paths, order='horizontal'):
    """
    Stacks images from the provided list of paths either horizontally or vertically.
    Resizes images to match the dimensions of the first image for consistency.
    """
    images = []
    for path in image_paths:
        img = load_image(path)
        images.append(img)

    if len(images) == 0:
        print("Error: No valid images found to stack.")
        return None

    # Use dimensions of the first image as reference
    h_ref, w_ref = images[0].shape[:2]

    processed_images = []
    for img in images:
        h, w = img.shape[:2]
        if order == 'horizontal':
            # Match height, scale width proportionally
            if h != h_ref:
                new_w = int(w * (h_ref / h))
                img_resized = cv2.resize(img, (new_w, h_ref))
                processed_images.append(img_resized)
            else:
                processed_images.append(img)
        else:
            # Match width, scale height proportionally
            if w != w_ref:
                new_h = int(h * (w_ref / w))
                img_resized = cv2.resize(img, (w_ref, new_h))
                processed_images.append(img_resized)
            else:
                processed_images.append(img)

    if order == 'horizontal':
        return np.hstack(processed_images)
    else:
        return np.vstack(processed_images)
import numpy as np
from pathlib import Path


def get_coins_sample(threshold=150, kernel_size=7):
    """
    Generate the 'coins' example. 
    Moved here to avoid hard dependency on skimage in the main library.
    """
    from skimage import data
    from scipy.ndimage import binary_dilation, binary_erosion
    from imtools.mask_utils import fill_holes_mask 

    img = data.coins()
    mask = img > threshold
    
    kernel = np.ones((kernel_size, kernel_size))
    mask_dilated = binary_dilation(mask, structure=kernel)
    mask_eroded = binary_erosion(mask_dilated, structure=kernel)
    mask = fill_holes_mask(mask_eroded)
    img3D = np.repeat(img[:, :, None], 3, axis=-1)

    return img3D, mask

def get_pedestrian_sample_yolo():
    """
    Generate the 'pedestrian' example using YOLO.
    
    Parameters
    ----------
    model_path : str
        Path to YOLO segmentation model weights.
    
    Returns
    -------
    tuple
        (image_path, masks) where masks is a boolean numpy array.
    """
    import os
    from ultralytics import YOLO
    from imtools import to_numpy_image
    
    # Use path relative to this file's location
    imgpath = 'assets/person_and_bike_188.png'
    assert os.path.exists(imgpath)
    
    model_path = 'checkpoint/yolo11n-seg.pt'
    model = YOLO(model_path)
    results = model.predict(source=str(imgpath), save=False, conf=0.5)[0]    
    img3D = to_numpy_image(imgpath, force_3d=True)
    return img3D, results


def get_pedestrian_sample_sam():
    from ultralytics.models.sam import SAM3SemanticPredictor
    import os
    import cv2
    from imtools import to_numpy_image
    
    
    # 1. Setup image and mask/result from SAM
    imgpath = 'assets/person_and_bike_188.png'
    assert os.path.exists(imgpath)
    
    img = cv2.imread(imgpath)
    img = img[:, :, ::-1].copy() # BGR to RGB
    
    # 2. Setup SAM predictor and run prediction
    # Get the BPE file path from current directory
    bpe_path = Path('checkpoint/bpe_simple_vocab_16e6.txt.gz')
    assert bpe_path.exists(), f"BPE file not found at {bpe_path}"
    
    # sam3 config
    conf_thresh = 0.4
    classes = ["person", "bus", "car", "glasses", "bicycle", "motorcycle"]
    #classes = ["person"]
    
    # Initialize predictor with configuration
    overrides = dict(
        conf=conf_thresh,
        task="segment",
        mode="predict",
        model="checkpoint/sam3.pt",
        half=True,  # Use FP16 for faster inference
        save=False,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides, bpe_path=str(bpe_path))
    
    # Query with multiple text prompts
    predictor.set_image(imgpath)
    results = predictor(text=classes)[0]    
    img3D = to_numpy_image(imgpath, force_3d=True)
    return img3D, results
    

    
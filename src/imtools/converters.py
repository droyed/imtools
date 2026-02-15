import numpy as np
import torch
import cv2


# --- PyTorch Backend Converters ---

def masks_to_label_image_torch_vectorized(masks_array, device=None):
    """
    Converts a stack of 3D masks to a 2D label image using PyTorch broadcasting.
    Optimized for speed on GPU.
    
    Args:
        masks_array (torch.Tensor or np.ndarray): Input masks (N, H, W).
        device (torch.device, optional): Target device. If None, preserves input device.
    """
    # 1. Prepare Tensor
    if isinstance(masks_array, torch.Tensor):
        if device is not None:
            masks_tensor = masks_array.to(device)
        else:
            masks_tensor = masks_array  # Keep existing device
    else:
        # If input is NumPy and device is None, defaults to CPU
        masks_tensor = torch.tensor(masks_array, device=device)
    
    # 2. Resolve Actual Device
    # We must use the tensor's actual device for the indices to avoid mismatches
    actual_device = masks_tensor.device
    
    n, h, w = masks_tensor.shape
    
    # 3. Create indices on the SAME device
    indices = torch.arange(1, n + 1, dtype=torch.int32, device=actual_device).view(n, 1, 1)
    
    # 4. Broadcast & Reduce
    # (N, H, W) * (N, 1, 1) -> (N, H, W). Max over dim 0 gives the highest index (label).
    label_image = (masks_tensor * indices).max(dim=0).values
    
    return label_image.cpu().numpy()


def masks_to_label_image_torch_loop(masks_array, device=None):
    """
    Converts a stack of 3D masks to a 2D label image using a PyTorch loop.
    Useful as a fallback if memory is constrained.
    """
    # 1. Prepare Tensor
    if isinstance(masks_array, torch.Tensor):
        if device is not None:
            masks = masks_array.to(device)
        else:
            masks = masks_array
    else:
        masks = torch.as_tensor(masks_array, device=device)

    actual_device = masks.device
    n, h, w = masks.shape
    
    # Create result tensor on the correct device
    label_image = torch.zeros((h, w), dtype=torch.int32, device=actual_device)
    
    for i in range(n):
        mask_slice = masks[i]
        if mask_slice.dtype != torch.bool:
             mask_slice = mask_slice.bool()
        label_image[mask_slice] = i + 1
        
    return label_image.cpu().numpy()


# --- NumPy Backend Converters ---

def masks_to_label_image_numpy_loop(masks_3d):
    """
    Converts a stack of 3D masks to a 2D label image using a NumPy loop.
    """
    if isinstance(masks_3d, torch.Tensor):
        masks_3d = masks_3d.detach().cpu().numpy()
        
    n_masks, h, w = masks_3d.shape
    label_image = np.zeros((h, w), dtype=np.int32)
    masks_bool = masks_3d.astype(bool, copy=False)
    
    for i in range(n_masks):
        label_image[masks_bool[i]] = i + 1
        
    return label_image


def masks_to_label_image_numpy_vectorized(masks_3d):
    """
    Converts a stack of 3D masks to a 2D label image using NumPy broadcasting.
    """
    if isinstance(masks_3d, torch.Tensor):
        masks_3d = masks_3d.detach().cpu().numpy()

    label_image = (masks_3d * np.arange(1, len(masks_3d)+1)[:, None, None]).max(axis=0)
    return label_image.astype(np.int32)


# --- Single Mask Converter ---

def binary_mask_to_label_image(mask, connectivity=8):
    """
    Converts a single binary mask into a label image using connected components.
    
    Args:
        mask (np.ndarray): Binary mask (H, W).
        connectivity (int): Pixel connectivity (4 or 8).
    
    Returns:
        np.ndarray: Label image where each connected component has a unique integer label.
    """
    # Ensure mask is uint8 for OpenCV
    mask_uint8 = mask.astype(np.uint8, copy=False)
    _, label_image = cv2.connectedComponents(mask_uint8, connectivity=connectivity)
    
    return label_image


# --- Main Dispatchers ---

def masks_to_label_image(masks_array, use_loop=True, use_numpy=True, device=None):
    """
    Main Dispatcher: Converts a stack of 3D masks (N, H, W) to a single 2D label image (H, W).
    
    Args:
        masks_array (np.ndarray or torch.Tensor): Input 3D masks.
        use_loop (bool): 
            - True: Use for-loop implementation (Default).
            - False: Use vectorized implementation.
        use_numpy (bool): 
            - True: Use NumPy backend (Default). 'device' arg is ignored.
            - False: Use PyTorch backend.
        device (str or None): 
            - None: Keep tensor on current device (or CPU if input is NumPy).
            - 'cpu': Force execution on CPU.
            - 'cuda': Force execution on GPU.
    """
    if use_numpy:
        # NumPy Backend
        if use_loop:
            return masks_to_label_image_numpy_loop(masks_array)
        else:
            return masks_to_label_image_numpy_vectorized(masks_array)
    else:
        # PyTorch Backend
        if use_loop:
            return masks_to_label_image_torch_loop(masks_array, device=device)
        else:
            return masks_to_label_image_torch_vectorized(masks_array, device=device)


def yolo_to_label_image(results, use_loop=True, use_numpy=None, device=None):
    """
    Converts YOLO results object to a 2D label image.
    Strictly expects boolean masks from the YOLO result object.
    
    Args:
        results: Ultralytics YOLO Results object.
        use_loop (bool): Use loop strategy.
        use_numpy (bool or None): 
            - None (Default): Automatically choose based on device.
            - True: Force NumPy backend.
            - False: Force PyTorch backend.
        device (str): Target device. Only applies if use_numpy is False.
    """
    # 1. Safety Check: Handle No Detections
    if results.masks is None:
        h, w = results.orig_shape
        # If use_numpy is None (auto), default to NumPy for empty results
        # Usually, returning a CPU numpy array is safest if we don't know the device.
        if use_numpy is True or use_numpy is None:
            return np.zeros((h, w), dtype=np.int32)
        else:
            target_device = device if device else 'cpu'
            return torch.zeros((h, w), device=target_device, dtype=torch.int32)

    # 2. Get 3D masks
    masks_tensor = results.masks.data

    # 2.1. Determine Backend Automatically
    if use_numpy is None:
        # If masks are on CPU, use NumPy (faster loop). 
        # If masks are on GPU, use PyTorch (avoid transfer).
        use_numpy = (masks_tensor.device.type == 'cpu')
            
    # 3. Conditional Conversion
    if masks_tensor.dtype != torch.bool:
        masks_tensor = masks_tensor.bool()

    # 4. Call Dispatcher
    return masks_to_label_image(
        masks_tensor, 
        use_loop=use_loop, 
        use_numpy=use_numpy, 
        device=device
    )
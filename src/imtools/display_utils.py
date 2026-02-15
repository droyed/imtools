import cv2
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional



def block_until_closed(window_name, close_key):
    # Convert close_key to key code
    if isinstance(close_key, str):
        if close_key.lower() == 'esc':
            key_code = 27  # ESC key
        else:
            key_code = ord(close_key[0])  # Get ASCII value of first character
    else:
        key_code = close_key
    
    # Wait for the specific key to be pressed
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == key_code:
            break
    
    # Clean up
    cv2.destroyWindow(window_name)

def get_screen_resolution():
    """Get the screen resolution on Linux using xrandr."""
    DEFAULT_RESOLUTION = (1920, 1080)
    
    try:
        output = subprocess.check_output(['xrandr']).decode('utf-8')
        for line in output.split('\n'):
            if ' connected' in line and 'primary' in line:
                resolution = line.split()[3]
                width = int(resolution.split('x')[0])
                height = int(resolution.split('x')[1].split('+')[0])
                return width, height
        return DEFAULT_RESOLUTION
    except (subprocess.CalledProcessError, IndexError, ValueError):
        return DEFAULT_RESOLUTION
    
def scale_image_to_fit(image, target_width, target_height, maintain_aspect=True):
    """
    Scale image to fit within target dimensions.
    
    Returns:
        Tuple of (scaled_image, x_offset, y_offset)
    """
    if not maintain_aspect:
        return cv2.resize(image, (target_width, target_height)), 0, 0
    
    img_height, img_width = image.shape[:2]
    scale = min(target_width / img_width, target_height / img_height)
    
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    
    resized_image = cv2.resize(image, (new_width, new_height))
    
    # Create black canvas
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Center the image
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image
    
    return canvas, x_offset, y_offset

def show_cv2_fullscreen(image, window_name='Fullscreen Image', maintain_aspect=True, 
                    show_borders=True, close_key='q', block=False, window_title=None):
    """
    Display image fullscreen on Linux, scaled to fill the entire screen.
    
    Parameters:
    -----------
    image : numpy.ndarray
        The image array to display
    window_name : str
        Name of the display window
    maintain_aspect : bool
        If True, maintains aspect ratio and centers the image
        If False, stretches image to fill screen
    show_borders : bool
        If True, shows window borders (maximized window)
        If False, hides borders (true fullscreen)
    close_key : str or int
        Key to close the window
    block : bool
        If True, blocks until the window is closed
        If False, returns immediately
    window_title : str, optional
        Optional window title. If provided, will be displayed with close instruction.
        If None, uses window_name with close instruction appended.
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Set window title with close instruction only when blocking
    if block:
        if window_title is None:
            close_key_str = 'ESC' if (isinstance(close_key, str) and close_key.lower() == 'esc') else f"'{close_key}'"
            window_title = f"{window_name} - Press {close_key_str} to close"
        else:
            close_key_str = 'ESC' if (isinstance(close_key, str) and close_key.lower() == 'esc') else f"'{close_key}'"
            window_title = f"{window_title} - Press {close_key_str} to close"
        cv2.setWindowTitle(window_name, window_title)
    
    screen_width, screen_height = get_screen_resolution()
    display_image, _, _ = scale_image_to_fit(image, screen_width, screen_height, maintain_aspect)
    print(f'⚠️ Warning: Image resolution is not the same as the screen resolution. The image is scaled to fit the screen. screen_width : {screen_width}, screen_height : {screen_height}')
    
    if show_borders:
        cv2.resizeWindow(window_name, screen_width, screen_height)
    else:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    cv2.imshow(window_name, display_image)
    
    if block:
        block_until_closed(window_name, close_key=close_key)

def show_cv2(image, window_name='Image', close_key='q', block=False, window_title=None):
    """
    Display image in a resizable window.
    
    Parameters:
    -----------
    image : numpy.ndarray
        The image array to display
    window_name : str
        Name of the display window (default: 'Image')
    close_key : str or int
        Key to close the window. Can be:
        - A character like 'q', 'x', 'c', etc.
        - 'esc' for Escape key
        - An integer key code (e.g., 27 for ESC)
    block : bool
        If True, blocks until the window is closed
        If False, returns immediately
    window_title : str, optional
        Optional window title. If provided, will be displayed with close instruction.
        If None, uses window_name with close instruction appended.
    """
    # Assert that image is a uint8 array
    if image.dtype != np.uint8:
        raise ValueError("Image must be a uint8 array")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Set window title with close instruction only when blocking
    if block:
        if window_title is None:
            close_key_str = 'ESC' if (isinstance(close_key, str) and close_key.lower() == 'esc') else f"'{close_key}'"
            window_title = f"{window_name} - Press {close_key_str} to close"
        else:
            close_key_str = 'ESC' if (isinstance(close_key, str) and close_key.lower() == 'esc') else f"'{close_key}'"
            window_title = f"{window_title} - Press {close_key_str} to close"
        cv2.setWindowTitle(window_name, window_title)
    
    cv2.imshow(window_name, image)

    if block:
        block_until_closed(window_name, close_key=close_key)

def show_mpl(
    image_array: np.ndarray,
    title: str = "Array Image",
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Optional[tuple] = None,
    show_colorbar: bool = False,
    normalize: bool = True,
    block: bool = False
) -> None:
    """
    Display an array image using matplotlib.
    
    Args:
        image_array: numpy array representing the image. Can be 2D (grayscale) or 3D (RGB/RGBA).
        title: title of the plot (default: "Array Image")
        cmap: colormap to use. If None, 'gray' is used for 2D arrays, None for 3D arrays.
        vmin: minimum value for colormap scaling. If None, uses array min.
        vmax: maximum value for colormap scaling. If None, uses array max.
        figsize: figure size tuple (width, height) in inches. If None, uses default.
        show_colorbar: whether to show colorbar (default: False)
        normalize: if True and array is integer type, normalize to [0, 1] range (default: True)
        block: whether to block execution until the window is closed (default: False)
    
    Example:
        >>> import numpy as np
        >>> arr = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        >>> display_array_image(arr)
    """
    if not isinstance(image_array, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(image_array)}")
    
    if image_array.size == 0:
        raise ValueError("Array is empty")
    
    # Handle integer arrays - normalize if requested
    if normalize and np.issubdtype(image_array.dtype, np.integer):
        array_min = image_array.min()
        array_max = image_array.max()
        if array_max > array_min:
            image_array = (image_array - array_min) / (array_max - array_min)
        else:
            image_array = image_array.astype(np.float64)
    
    # Determine colormap
    if cmap is None:
        if image_array.ndim == 2:
            cmap = 'gray'
        else:
            cmap = None
    
    # Set up figure
    if figsize is not None:
        plt.figure(figsize=figsize)
    else:
        plt.figure()
    
    # Display image
    im = plt.imshow(image_array, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis('off')
    
    # Add colorbar if requested
    if show_colorbar:
        plt.colorbar(im)
    
    plt.tight_layout()
    plt.show(block=block)


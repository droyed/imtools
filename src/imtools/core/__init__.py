"""Core subpackage for imtools.

Re-exports the primary data structures and image-format conversion
utilities used throughout the package.

Public API:
    - :class:`~imtools.core.types.Annotation`
    - :class:`~imtools.core.types.BlendConfig`
    - :class:`~imtools.core.types.TitleConfig`
    - :class:`~imtools.core.types.LabelStyle`
    - :func:`~imtools.core.formats.pil_to_opencv`
    - :func:`~imtools.core.formats.opencv_to_pil`
    - :func:`~imtools.core.formats.to_pil_image`
    - :func:`~imtools.core.formats.to_numpy_image`
    - :func:`~imtools.core.formats.imwrite`
"""

from imtools.core.types import Annotation, BlendConfig, TitleConfig, LabelStyle
from imtools.core.formats import pil_to_opencv, opencv_to_pil, to_pil_image, to_numpy_image, imwrite

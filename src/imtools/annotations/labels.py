"""Label formatting system for annotation overlays.

Provides :class:`LabelFormat` (40+ format options), :class:`LabelContext`
(detection metadata container), and :func:`resolve_label` (dispatcher) for
generating human-readable annotation strings.
"""

from __future__ import annotations

import json
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional, Union, Callable, Dict

class LabelFormat(str, Enum):
    """Enumeration of available annotation label format styles.

    Merges standard YOLO tracking formats with scientific/geometric formats.
    Each member maps to a specific string-rendering strategy in
    :func:`resolve_label`.

    Members are grouped by category:

    - **Standard** — class name with confidence score.
    - **Geometric** — includes bounding-box dimensions or aspect ratio.
    - **Positional** — includes centroid coordinates.
    - **Indexing** — tracking-ID focused formats.
    - **Data** — machine-readable (JSON, CSV).
    - **Pipe-separated** — compact single-line with ``|`` delimiters.
    - **Multi-line** — stacked two- or three-line labels.
    - **Scientific** — confidence in scientific notation.

    Example:
        >>> from imtools.annotations.labels import LabelFormat, LabelContext, resolve_label
        >>> ctx = LabelContext(class_name='car', conf=0.87)
        >>> resolve_label(LabelFormat.DEFAULT, ctx)
        'car 0.87'
    """
    # --- 1. Standard ---
    DEFAULT = "default"                # "person 0.95"
    CONFIDENCE_ONLY = "conf_only"      # "95%"
    CLASS_ONLY = "class_only"          # "person"
    CLASS_HASHINDEX = "class_hashindex"   # "person #42"
    INDEXED = "indexed"                # "person_42 0.95"

    # --- 2. Geometric / Physical ---
    DIMENSIONS_2D = "dims_2d"          # "person 0.95 [100x200]" (WxH)
    ASPECT_RATIO = "aspect_ratio"      # "person | AR: 0.5"
    AREA_PIXELS = "area_px"            # "person | 2500px²"
    DETAILED = "detailed"              # "person 0.95 A:2500"

    # --- 3. Positional ---
    COORDS_INT = "coords_int"          # "person @ 100,200"
    COORDS = "coords"                  # "person (100,200)"

    # --- 4. Indexing (Tracking/Counting) ---
    HASHINDEX_ONLY = "hashindex_only"          # "#42"
    INDEX_ONLY = "index_only"        # "42"
    INDEX_MINIMAL = "index_minimal"    # "#42 0.95"
    HASHINDEX_CONF = "hashindex_conf"         # "#42 | 0.95"
    INDEX_HASH = "index_hash"          # "#42 person"
    INDEX_CIRCLE = "index_circle"      # "(42) person 0.95"

    # --- 5. Data / Machine Readable ---
    JSON_STYLE = "json_style"          # {"cls": "person", "conf": 0.95}
    CSV_STYLE = "csv_style"            # person,0.95,42,100,200

    # --- 6. Pipe-Separated (One Line) ---
    PIPE_BASIC = "pipe_basic"          # "person | 0.95"
    PIPE_INDEXED = "pipe_indexed"      # "#42 | person | 0.95"
    PIPE_COORDS = "pipe_coords"        # "person | 0.95 | (100,200)"
    PIPE_AREA = "pipe_area"            # "person | 0.95 | A:2500"
    PIPE_FULL = "pipe_full"            # "#42 | person | 0.95 | 120x400"
    PIPE_FULL_AREA = "pipe_full_area"  # "person | 0.95 | A:2500 | (100,200)"

    # --- 7. Multi-line (Stacked) ---
    MULTI_BASIC = "multi_basic"        # "person\n0.95"
    MULTI_BBOX = "multi_bbox"          # "person 0.95\nW:120 H:400"
    MULTI_COORDS = "multi_coords"      # "person 0.95\n(100,200)"
    MULTI_AREA = "multi_area"          # "person 0.95\nA:2500 px"
    MULTI_AREA_COORDS = "multi_area_coords" # "person 0.95\nA:2500 | (100,200)"
    MULTI_FULL = "multi_full"          # "person 0.95\nA:2500 px\n(100,200)"
    MULTI_DETAILED = "multi_detailed"  # "#42 person\n0.95\n100x200"
    MULTI_DETAILED_AREA = "multi_detailed_area" # "person_42 0.95\nA:2500 | P:200\n(100,200)"

    # --- 8. Scientific ---
    SCIENTIFIC = "scientific"          # "person 9.50e-01"


@dataclass
class LabelContext:
    """Container holding all data available for a single detection.

    Fields are optional to support cases where tracking IDs or segmentation
    properties are not available.

    Attributes:
        class_name: Human-readable class/category name (e.g. ``'person'``).
        conf: Confidence score in ``[0.0, 1.0]``.
        index: Optional tracking or sequence ID (e.g. YOLO track ID).
        cx: Centroid x-coordinate (column index) in image pixels.
        cy: Centroid y-coordinate (row index) in image pixels.
        width: Bounding-box width in pixels.
        height: Bounding-box height in pixels.
        area: Mask area in pixels; ``None`` if unavailable.
        perimeter: Mask perimeter in pixels; ``None`` if unavailable.

    Example:
        >>> ctx = LabelContext(class_name='dog', conf=0.93, index=7,
        ...                    cx=320, cy=240, width=80, height=120)
        >>> ctx.pct
        '93%'
        >>> ctx.hash_idx
        '#7'
    """
    class_name: str
    conf: float
    index: Optional[int] = None       # Tracking ID
    cx: int = 0                       # Center X
    cy: int = 0                       # Center Y
    width: int = 0                    # Bounding Box Width
    height: int = 0                   # Bounding Box Height
    area: Optional[float] = None      # Mask Area (pixels)
    perimeter: Optional[float] = None # Mask Perimeter (pixels)

    # --- Helpers for concise formatting ---
    @property
    def pct(self) -> str:
        """Confidence formatted as a percentage string (e.g. ``'93%'``)."""
        return f"{self.conf:.0%}"

    @property
    def flt(self) -> str:
        """Confidence formatted as a two-decimal float string (e.g. ``'0.93'``)."""
        return f"{self.conf:.2f}"

    @property
    def sci(self) -> str:
        """Confidence in scientific notation (e.g. ``'9.30e-01'``)."""
        return f"{self.conf:.2e}"

    @property
    def idx(self) -> str:
        """Index as a string, or ``'?'`` if :attr:`index` is ``None``."""
        return str(self.index) if self.index is not None else "?"

    @property
    def hash_idx(self) -> str:
        """Index prefixed with ``#`` (e.g. ``'#7'``), or ``'#?'`` if absent."""
        return f"#{self.idx}"

    @property
    def safe_area(self) -> str:
        """Integer area string, or ``'?'`` if :attr:`area` is ``None``."""
        return f"{int(self.area)}" if self.area is not None else "?"

    @property
    def safe_perim(self) -> str:
        """Integer perimeter string, or ``'?'`` if :attr:`perimeter` is ``None``."""
        return f"{int(self.perimeter)}" if self.perimeter is not None else "?"

    @property
    def dims(self) -> str:
        """Bounding-box dimensions as ``'WxH'`` string."""
        return f"{self.width}x{self.height}"

    @property
    def coords(self) -> str:
        """Centroid as ``'(cx,cy)'`` string."""
        return f"({self.cx},{self.cy})"


# --- Helper for Complex Formats (JSON) ---
def _fmt_json(c: LabelContext) -> str:
    """Render a :class:`LabelContext` as a compact JSON object string."""
    data = {
        "cls": c.class_name, "conf": round(c.conf, 2),
        "id": c.index, "cx": c.cx, "cy": c.cy, "area": c.area
    }
    return json.dumps({k: v for k, v in data.items() if v is not None})


# --- Dispatch Table ---
# Maps Enum members to callables that take a LabelContext and return a string
_FORMAT_DISPATCH: Dict[LabelFormat, Callable[[LabelContext], str]] = {
    # 1. Standard
    LabelFormat.DEFAULT:          lambda c: f"{c.class_name} {c.flt}",
    LabelFormat.CONFIDENCE_ONLY:  lambda c: c.pct,
    LabelFormat.CLASS_ONLY:       lambda c: c.class_name,
    LabelFormat.CLASS_HASHINDEX: lambda c: f"{c.class_name} {c.hash_idx}",
    LabelFormat.INDEXED:          lambda c: f"{c.class_name}_{c.idx} {c.flt}",

    # 2. Geometric
    LabelFormat.DIMENSIONS_2D:    lambda c: f"{c.class_name} {c.flt} [{c.dims}]",
    LabelFormat.ASPECT_RATIO:     lambda c: f"{c.class_name} | AR: {(c.width/c.height if c.height else 0):.2f}",
    LabelFormat.AREA_PIXELS:      lambda c: f"{c.class_name} | {c.safe_area}px²",
    LabelFormat.DETAILED:         lambda c: f"{c.class_name} {c.flt} A:{c.safe_area}",

    # 3. Positional
    LabelFormat.COORDS_INT:       lambda c: f"{c.class_name} @ {c.cx},{c.cy}",
    LabelFormat.COORDS:           lambda c: f"{c.class_name} {c.coords}",

    # 4. Indexing
    LabelFormat.HASHINDEX_ONLY:   lambda c: c.hash_idx,
    LabelFormat.INDEX_ONLY:       lambda c: c.idx,
    LabelFormat.INDEX_MINIMAL:    lambda c: f"{c.hash_idx} {c.flt}",
    LabelFormat.HASHINDEX_CONF:      lambda c: f"{c.hash_idx} | {c.flt}",
    LabelFormat.INDEX_HASH:       lambda c: f"{c.hash_idx} {c.class_name}",
    LabelFormat.INDEX_CIRCLE:     lambda c: f"({c.idx}) {c.class_name} {c.flt}",

    # 5. Data
    LabelFormat.JSON_STYLE:       _fmt_json,
    LabelFormat.CSV_STYLE:        lambda c: f"{c.class_name},{c.flt},{c.idx},{c.cx},{c.cy}",

    # 6. Pipe-Separated
    LabelFormat.PIPE_BASIC:       lambda c: f"{c.class_name} | {c.flt}",
    LabelFormat.PIPE_INDEXED:     lambda c: f"{c.hash_idx} | {c.class_name} | {c.flt}",
    LabelFormat.PIPE_COORDS:      lambda c: f"{c.class_name} | {c.flt} | {c.coords}",
    LabelFormat.PIPE_AREA:        lambda c: f"{c.class_name} | {c.flt} | A:{c.safe_area}",
    LabelFormat.PIPE_FULL:        lambda c: f"{c.hash_idx} | {c.class_name} | {c.flt} | {c.dims}",
    LabelFormat.PIPE_FULL_AREA:   lambda c: f"{c.class_name} | {c.flt} | A:{c.safe_area} | {c.coords}",

    # 7. Multi-line
    LabelFormat.MULTI_BASIC:          lambda c: f"{c.class_name}\n{c.flt}",
    LabelFormat.MULTI_BBOX:           lambda c: f"{c.class_name} {c.flt}\nW:{c.width} H:{c.height}",
    LabelFormat.MULTI_COORDS:         lambda c: f"{c.class_name} {c.flt}\n{c.coords}",
    LabelFormat.MULTI_AREA:           lambda c: f"{c.class_name} {c.flt}\nA:{c.safe_area} px",
    LabelFormat.MULTI_AREA_COORDS:    lambda c: f"{c.class_name} {c.flt}\nA:{c.safe_area} | {c.coords}",
    LabelFormat.MULTI_FULL:           lambda c: f"{c.class_name} {c.flt}\nA:{c.safe_area} px\n{c.coords}",
    LabelFormat.MULTI_DETAILED:       lambda c: f"{c.hash_idx} {c.class_name}\n{c.pct}\n{c.dims}",
    LabelFormat.MULTI_DETAILED_AREA:  lambda c: f"{c.class_name}_{c.idx} {c.flt}\nA:{c.safe_area} | P:{c.safe_perim}\n{c.coords}",

    # 8. Scientific
    LabelFormat.SCIENTIFIC:       lambda c: f"{c.class_name} {c.sci}",
}


def resolve_label(fmt: Union[LabelFormat, str], ctx: LabelContext) -> str:
    """Generate a formatted annotation string for a detection.

    Uses a dispatch table for all known :class:`LabelFormat` members.
    Falls back to Python's :meth:`str.format` for custom template strings,
    exposing every :class:`LabelContext` field as a named placeholder.

    Args:
        fmt: A :class:`LabelFormat` enum member **or** a custom Python
            format-string template (e.g.
            ``"{class_name} detected at ({cx},{cy})"``).
        ctx: :class:`LabelContext` instance supplying the detection data.

    Returns:
        Formatted label string.  Falls back to ``"{class_name} {conf:.2f}"``
        if the format string is invalid or contains unknown keys.

    Example:
        >>> ctx = LabelContext(class_name='bus', conf=0.72, cx=400, cy=300)
        >>> resolve_label(LabelFormat.PIPE_BASIC, ctx)
        'bus | 0.72'
        >>> resolve_label("{class_name} @ ({cx},{cy})", ctx)
        'bus @ (400,300)'
    """
    # 1. Try efficient Dispatch Table lookup
    if isinstance(fmt, LabelFormat):
        handler = _FORMAT_DISPATCH.get(fmt)
        if handler:
            return handler(ctx)

    # 2. Fallback: Custom Template string (Power User Feature)
    try:
        # Using asdict allows the format string to use any field name
        # e.g., "{class_name} found at {cx},{cy}"
        return str(fmt).format(**asdict(ctx))
    except (KeyError, ValueError, AttributeError):
        # 3. Ultimate Fallback
        return f"{ctx.class_name} {ctx.flt}"

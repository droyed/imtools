import json
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional, Union, Callable, Dict

class LabelFormat(str, Enum):
    """
    Enum for label format options.
    Merges standard YOLO tracking formats with scientific/geometric analysis formats.
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
    """
    Holds all data available for a specific detection for string formatting.
    Fields are Optional to handle cases where tracking (index) or segmentation (area) 
    might not be active.
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
        return f"{self.conf:.0%}"
    
    @property
    def flt(self) -> str: 
        return f"{self.conf:.2f}"
    
    @property
    def sci(self) -> str: 
        return f"{self.conf:.2e}"
    
    @property
    def idx(self) -> str: 
        return str(self.index) if self.index is not None else "?"
    
    @property
    def hash_idx(self) -> str: 
        return f"#{self.idx}"
    
    @property
    def safe_area(self) -> str: 
        return f"{int(self.area)}" if self.area is not None else "?"
    
    @property
    def safe_perim(self) -> str: 
        return f"{int(self.perimeter)}" if self.perimeter is not None else "?"
    
    @property
    def dims(self) -> str: 
        return f"{self.width}x{self.height}"
    
    @property
    def coords(self) -> str: 
        return f"({self.cx},{self.cy})"


# --- Helper for Complex Formats (JSON) ---
def _fmt_json(c: LabelContext) -> str:
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
    """
    Generates the text string based on the format and context using a dispatch table.
    
    Args:
        fmt: LabelFormat enum value or custom format string
        ctx: LabelContext containing detection data
        
    Returns:
        Formatted label string
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
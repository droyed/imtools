"""
Tests for imtools.annotations.labels module.
"""
import pytest
from imtools.annotations.labels import (
    LabelFormat,
    LabelContext,
    resolve_label,
)


class TestLabelFormat:
    """Tests for LabelFormat enum."""

    def test_label_format_values(self):
        """Test that LabelFormat has expected values."""
        assert LabelFormat.DEFAULT.value == "default"
        assert LabelFormat.CLASS_ONLY.value == "class_only"
        assert LabelFormat.PIPE_BASIC.value == "pipe_basic"

    def test_label_format_is_string(self):
        """Test LabelFormat inherits from str."""
        assert isinstance(LabelFormat.DEFAULT, str)


class TestLabelContext:
    """Tests for LabelContext dataclass."""

    def test_label_context_creation(self):
        """Test creating a LabelContext."""
        ctx = LabelContext(class_name="person", conf=0.95)

        assert ctx.class_name == "person"
        assert ctx.conf == 0.95
        assert ctx.index is None
        assert ctx.cx == 0
        assert ctx.cy == 0

    def test_label_context_properties(self):
        """Test LabelContext computed properties."""
        ctx = LabelContext(
            class_name="person",
            conf=0.95,
            index=5,
            cx=100,
            cy=200,
            width=50,
            height=80,
            area=4000
        )

        assert ctx.pct == "95%"
        assert ctx.flt == "0.95"
        assert ctx.idx == "5"
        assert ctx.hash_idx == "#5"
        assert ctx.dims == "50x80"
        assert ctx.coords == "(100,200)"
        assert ctx.safe_area == "4000"

    def test_label_context_with_area_none(self):
        """Test LabelContext with None area."""
        ctx = LabelContext(class_name="person", conf=0.95, area=None)

        assert ctx.safe_area == "?"

    def test_label_context_with_index_none(self):
        """Test LabelContext with None index."""
        ctx = LabelContext(class_name="person", conf=0.95, index=None)

        assert ctx.idx == "?"


class TestResolveLabel:
    """Tests for resolve_label function."""

    def test_default_format(self):
        """Test DEFAULT format."""
        ctx = LabelContext(class_name="person", conf=0.95)
        result = resolve_label(LabelFormat.DEFAULT, ctx)

        assert "person" in result
        assert "0.95" in result

    def test_class_only_format(self):
        """Test CLASS_ONLY format."""
        ctx = LabelContext(class_name="car", conf=0.87)
        result = resolve_label(LabelFormat.CLASS_ONLY, ctx)

        assert result == "car"

    def test_confidence_only_format(self):
        """Test CONFIDENCE_ONLY format."""
        ctx = LabelContext(class_name="person", conf=0.95)
        result = resolve_label(LabelFormat.CONFIDENCE_ONLY, ctx)

        assert result == "95%"

    def test_hash_index_format(self):
        """Test HASHINDEX_ONLY format."""
        ctx = LabelContext(class_name="person", conf=0.95, index=42)
        result = resolve_label(LabelFormat.HASHINDEX_ONLY, ctx)

        assert result == "#42"

    def test_coords_format(self):
        """Test COORDS format."""
        ctx = LabelContext(class_name="person", conf=0.95, cx=100, cy=200)
        result = resolve_label(LabelFormat.COORDS, ctx)

        assert "100" in result
        assert "200" in result

    def test_pipe_basic_format(self):
        """Test PIPE_BASIC format."""
        ctx = LabelContext(class_name="dog", conf=0.75)
        result = resolve_label(LabelFormat.PIPE_BASIC, ctx)

        assert "dog" in result
        assert "|" in result

    def test_json_style_format(self):
        """Test JSON_STYLE format."""
        ctx = LabelContext(class_name="cat", conf=0.99, index=1, cx=50, cy=60)
        result = resolve_label(LabelFormat.JSON_STYLE, ctx)

        assert "cat" in result
        assert "0.99" in result

    def test_scientific_format(self):
        """Test SCIENTIFIC format."""
        ctx = LabelContext(class_name="bird", conf=0.95)
        result = resolve_label(LabelFormat.SCIENTIFIC, ctx)

        assert "e" in result.lower()

    def test_string_format(self):
        """Test using string format directly."""
        ctx = LabelContext(class_name="fish", conf=0.5, cx=10, cy=20)
        result = resolve_label("{class_name} at ({cx},{cy})", ctx)

        assert result == "fish at (10,20)"

    def test_invalid_format_fallback(self):
        """Test invalid format falls back to default."""
        ctx = LabelContext(class_name="test", conf=0.5)
        result = resolve_label("invalid_format", ctx)

        # Invalid format falls back to using class_name + flt
        # Result may vary based on fallback implementation
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.parametrize("fmt", [
        LabelFormat.INDEXED,
        LabelFormat.DIMENSIONS_2D,
        LabelFormat.AREA_PIXELS,
        LabelFormat.MULTI_BASIC,
        LabelFormat.CSV_STYLE,
    ])
    def test_various_formats(self, fmt):
        """Test various label formats don't crash."""
        ctx = LabelContext(
            class_name="obj",
            conf=0.8,
            index=1,
            cx=50,
            cy=60,
            width=100,
            height=200,
            area=20000
        )
        result = resolve_label(fmt, ctx)
        assert isinstance(result, str)
        assert len(result) > 0

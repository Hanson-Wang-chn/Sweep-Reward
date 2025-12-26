# Metrics submodule
# Lazy imports to avoid loading torch when not needed


def __getattr__(name):
    """Lazy import to avoid loading heavy dependencies until needed."""
    if name == "GeometricMetrics":
        from .geometric import GeometricMetrics
        return GeometricMetrics
    elif name == "ContourMetrics":
        from .contour import ContourMetrics
        return ContourMetrics
    elif name == "SemanticMetrics":
        from .semantic import SemanticMetrics
        return SemanticMetrics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["GeometricMetrics", "ContourMetrics", "SemanticMetrics"]

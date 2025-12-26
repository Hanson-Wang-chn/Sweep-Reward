# Sweep-Reward Multi-modal Ensemble Evaluation Module
# Lazy imports to avoid loading torch when not needed


def __getattr__(name):
    """Lazy import to avoid loading heavy dependencies until needed."""
    if name == "Preprocessor":
        from .preprocessor import Preprocessor
        return Preprocessor
    elif name == "Evaluator":
        from .evaluator import Evaluator
        return Evaluator
    elif name == "VLMClient":
        from .vlm_client import VLMClient
        return VLMClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Preprocessor", "Evaluator", "VLMClient"]

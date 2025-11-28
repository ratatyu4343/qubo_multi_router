from .Objects import GridBoard, Net, Path, Pin
from .utilities import path_cost, path_to_str, path_turns

# Optional: MultiNetQuantumRouter may require optional dependencies (dimod, neal, etc.)
try:
    from .MultiNetQuantumRouter import MultiNetQuantumRouter  # type: ignore
except Exception as _e:  # pragma: no cover
    # Provide a stub that raises a clear error upon use, while keeping import surface stable
    class MultiNetQuantumRouter:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "MultiNetQuantumRouter is unavailable due to missing optional dependencies: "
                + str(_e)
            )

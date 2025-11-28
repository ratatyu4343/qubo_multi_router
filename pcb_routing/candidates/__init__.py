from typing import Dict, Type

from .astar import AStarGenerator

REGISTRY: Dict[str, Type] = {
    'astar': AStarGenerator,
}

def register(name: str, cls: Type) -> None:
    REGISTRY[name] = cls


"""
Compatibility shim for candidate generators.

All generators have been moved to pcb_routing.candidates.* modules.
This file re-exports those classes so any legacy imports continue to work.
"""

from .candidates.base import CandidateGenerator
from .candidates.astar import AStarGenerator

__all__ = [
    'CandidateGenerator',
    'AStarGenerator',
]

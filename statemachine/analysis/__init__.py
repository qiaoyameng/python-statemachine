"""Structural analysis of statecharts.

This package provides :func:`analyze_structure`, a *static* analyzer that
inspects a state machine class without instantiating it and without running
any user-defined guard, validator, action, entry or exit callback.
"""

from .structural import Certainty
from .structural import Finding
from .structural import FindingKind
from .structural import Path
from .structural import PathStep
from .structural import StructureReport
from .structural import analyze_structure

__all__ = [
    "Certainty",
    "Finding",
    "FindingKind",
    "Path",
    "PathStep",
    "StructureReport",
    "analyze_structure",
]

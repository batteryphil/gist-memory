"""
Gist Memory: Biomimetic Constructive Associative Memory for Language Models
===========================================================================
"""

from .core import (
    GistLayer,
    GenerativeThoughtReconstructionLayer,
    RMSNorm,
)
from .multihead import MultiHeadGistLayer
from .chunked import ChunkedGistLayer
from .state import GistState
from .decay import MultiScaleDecay, DataDependentDecay
from .adapter import (
    GistWrapperLayer,
    GistModelAdapter,
    GistCache,
)
from .swarm import (
    SwarmAgent,
    SwarmMemoryPool,
    GistSwarm,
    SwarmTelemetry,
)

__version__ = "0.1.0"

__all__ = [
    "GistLayer",
    "GenerativeThoughtReconstructionLayer",
    "MultiHeadGistLayer",
    "ChunkedGistLayer",
    "GistState",
    "MultiScaleDecay",
    "DataDependentDecay",
    "GistWrapperLayer",
    "GistModelAdapter",
    "GistCache",
    "SwarmAgent",
    "SwarmMemoryPool",
    "GistSwarm",
    "SwarmTelemetry",
]


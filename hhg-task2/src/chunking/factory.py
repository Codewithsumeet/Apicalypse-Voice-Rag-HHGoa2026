"""
Chunking strategy factory.

Provides a single entry point to instantiate any chunking strategy by name.
"""

from src.chunking.base import BaseChunker
from src.chunking.fixed_size import FixedSizeChunker
from src.chunking.semantic import SemanticChunker
from src.chunking.metadata_aware import MetadataAwareChunker


STRATEGY_REGISTRY: dict[str, type[BaseChunker]] = {
    "fixed": FixedSizeChunker,
    "semantic": SemanticChunker,
    "metadata_aware": MetadataAwareChunker,
}


def get_chunker(strategy: str = "fixed", **kwargs) -> BaseChunker:
    """
    Factory function to get a chunker by strategy name.

    Args:
        strategy: One of 'fixed', 'semantic', 'metadata_aware'.
        **kwargs: Strategy-specific configuration parameters.

    Returns:
        A configured BaseChunker instance.

    Raises:
        ValueError: If the strategy name is not recognized.
    """
    if strategy not in STRATEGY_REGISTRY:
        available = ", ".join(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Available: {available}")

    return STRATEGY_REGISTRY[strategy](**kwargs)


def list_strategies() -> list[str]:
    """Return list of available strategy names."""
    return list(STRATEGY_REGISTRY.keys())

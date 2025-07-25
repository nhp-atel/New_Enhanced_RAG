"""Performance tests placeholder.

This file exists to prevent CI failures when performance tests are run.
Replace with actual benchmarks when concrete implementations are available.
"""

import pytest


def test_performance_placeholder(benchmark):
    """Placeholder benchmark test to prevent empty test suite failure."""
    
    def sample_function():
        return sum(range(100))
    
    result = benchmark(sample_function)
    assert result == 4950


@pytest.mark.skip(reason="Placeholder - requires concrete implementations")
def test_chunking_performance(benchmark):
    """Placeholder for chunking performance benchmark."""
    pass


@pytest.mark.skip(reason="Placeholder - requires concrete implementations") 
def test_embedding_performance(benchmark):
    """Placeholder for embedding performance benchmark."""
    pass


@pytest.mark.skip(reason="Placeholder - requires concrete implementations")
def test_query_performance(benchmark):
    """Placeholder for query performance benchmark."""
    pass
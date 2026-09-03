"""Tests for the gap-crossing analysis module layout."""

import importlib
import unittest


class MemoryAnalysisModuleLayoutTests(unittest.TestCase):
    """Check the public module path for memory analyses."""

    def test_entropy_analysis_imports_from_memory_analysis_package(self):
        """The entropy analysis remains importable after reorganization."""
        entropy_analysis = importlib.import_module(
            "gap_crossing.memory_analysis.gap_cross_db_entropy"
        )

        self.assertTrue(callable(entropy_analysis.run))

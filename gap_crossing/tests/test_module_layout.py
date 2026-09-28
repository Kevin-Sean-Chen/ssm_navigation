"""Tests for the gap-crossing analysis module layout."""

import importlib
from pathlib import Path
import subprocess
import sys
import unittest


class MemoryAnalysisModuleLayoutTests(unittest.TestCase):
    """Check the public module path for memory analyses."""

    def test_entropy_analysis_imports_from_memory_analysis_package(self):
        """The entropy analysis remains importable after reorganization."""
        entropy_analysis = importlib.import_module(
            "gap_crossing.memory_analysis.gap_cross_db_entropy"
        )

        self.assertTrue(callable(entropy_analysis.run))

    def test_memory_module_imports_from_repository_root(self):
        """The memory module imports in a clean Python process."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import gap_crossing.memory_analysis.gap_cross_db_memory",
            ],
            cwd=Path(__file__).parents[2],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

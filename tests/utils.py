"""Module containing some utilities for testing."""

from pathlib import Path
import pkg_resources as pkg

__all__ = ["PATH_DEEPRANK_GNN", "PATH_TEST"]

# Environment data
PATH_DEEPRANK_GNN = Path(pkg.resource_filename('deeprank_gnn', ''))
ROOT = PATH_DEEPRANK_GNN.parent

PATH_TEST = ROOT / "tests"

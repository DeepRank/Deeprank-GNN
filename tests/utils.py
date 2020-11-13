"""Module containing some utilities for testing."""

from pathlib import Path
import pkg_resources as pkg

__all__ = ["PATH_GRAPHPROT", "PATH_TEST"]

# Environment data
PATH_GRAPHPROT = Path(pkg.resource_filename('graphprot', ''))
ROOT = PATH_GRAPHPROT.parent

PATH_TEST = ROOT / "tests"

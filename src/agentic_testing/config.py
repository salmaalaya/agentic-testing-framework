"""
Project-wide paths.

Every module resolves corpus/, reports/ and screenshots/ through this file
instead of walking up from its own location, so the package can live under
src/ while the data directories stay at the repository root.

The root can be overridden with the AGENTIC_TESTING_ROOT environment
variable (useful when the package is installed outside the repository).
"""

import os
import sys
from pathlib import Path


def _find_project_root() -> Path:
    env_root = os.getenv("AGENTIC_TESTING_ROOT")
    if env_root:
        return Path(env_root).resolve()
    # src/agentic_testing/config.py -> repository root is two levels above src/
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "corpus").is_dir():
        return repo_root
    return Path.cwd()


PROJECT_ROOT = _find_project_root()
CORPUS_DIR = PROJECT_ROOT / "corpus"
REPORTS_DIR = PROJECT_ROOT / "reports"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"


def ensure_project_root_on_path() -> None:
    """Make top-level folders (target_app/, scripts/) importable from the package."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def relative_to_root(path: Path) -> str:
    """Return a POSIX path relative to the project root (no user-specific absolute paths in reports)."""
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(path).name

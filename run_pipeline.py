"""
run_pipeline.py — Command-line entry point of the agentic testing pipeline.

Thin wrapper around agentic_testing.pipeline.main() so the pipeline can be
launched from the repository root without installing the package:

    python run_pipeline.py --skip-generator --no-check
    python run_pipeline.py --canal all --skip-generator --no-check --dashboard --pdf

See `python run_pipeline.py --help` for every option.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from agentic_testing.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()

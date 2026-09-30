"""Backward-compatible entry point (``python -m readme_rosetta.rosetta``)."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())

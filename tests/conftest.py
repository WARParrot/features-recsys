"""Make the src-layout package importable when tests run from a checkout."""

import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

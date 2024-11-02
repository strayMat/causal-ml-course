import os
from pathlib import Path

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

ROOT_DIR = Path(
    os.getenv("ROOT_DIR", Path(os.path.dirname(os.path.abspath(__file__))) / "..")
)

# Default paths
DIR2DATA = ROOT_DIR / "data"
DIR2RESULTS = DIR2DATA / "results"

DIR2DOCS = ROOT_DIR / "docs/source"
DIR2DOCS_STATIC = DIR2DOCS / "_static"

"""Put `src/` on sys.path so the scripts run without an install step."""
import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parent
ROOT = SRC.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

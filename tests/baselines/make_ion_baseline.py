import json
from pathlib import Path
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.e2e.utils.ion_validation import derive_limits_from_csv

CSV = Path("/Users/gukil/CherryAI/CherryAI_0717/ion_implant_3lot_dataset.csv")
OUT = Path("tests/baselines/ion_implant_limits.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

derived = derive_limits_from_csv(CSV)
if not derived:
    raise SystemExit("No TW-like column found in CSV")
mu, low, high = derived
OUT.write_text(json.dumps({"tw_avg":mu, "low":low, "high":high}, ensure_ascii=False, indent=2), encoding="utf-8")
print("Baseline saved:", OUT)
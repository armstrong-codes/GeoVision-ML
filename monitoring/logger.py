
import json
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOG_DIR = PROJECT_ROOT / "results"
LOG_FILE = LOG_DIR / "prediction_logs.jsonl"


def log_prediction(
    filename,
    predicted_class,
    confidence,
    response_time
):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "response_time_ms": float(response_time)
    }

    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def read_prediction_logs():

    if not LOG_FILE.exists():
        return []

    records = []

    with LOG_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return records

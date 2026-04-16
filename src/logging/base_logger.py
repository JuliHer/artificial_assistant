import json
import os
import time

LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)

class BaseLogger:
    def __init__(self, filename):
        self.path =  os.path.join(LOG_DIR, filename)

    def _write(self, record: dict):
        record["timestamp"] = time.time()

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
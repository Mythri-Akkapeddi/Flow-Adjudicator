"""JSONL trace writer: every LLM call, tool call and the final result."""
import json
import time
from pathlib import Path


class Tracer:
    def __init__(self, path, **meta):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w")
        self._t0 = time.time()
        self.log("start", **meta)

    def log(self, event: str, **data):
        rec = {"t": round(time.time() - self._t0, 3), "event": event, **data}
        self._f.write(json.dumps(rec, default=str) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


class NullTracer:
    def log(self, event: str, **data):
        pass

    def close(self):
        pass
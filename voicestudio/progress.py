"""Progress for long jobs. The job writes a small JSON file; the page reads it to draw a bar with a time estimate."""
import json, os, time
from .config import CFG


class Progress:
    def __init__(self, name, kind):
        self.path = CFG.runs / name / f"{kind}.progress.json"; self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stage = None; self.t0 = time.time(); self.t_start = time.time()

    def update(self, stage, done, total, note="", frac=None):
        """frac (0..1) = how far the whole job is, for jobs that are one big step (e.g. a single long recording): the bar and estimate follow it."""
        if stage != self.stage: self.stage, self.t0 = stage, time.time()
        eta = None
        if frac is not None:
            if frac >= 0.03 and frac < 1: eta = round((time.time() - self.t_start) / frac * (1 - frac))
        elif done >= 1 and total > done: eta = round((time.time() - self.t0) / done * (total - done))
        self._write({"stage": stage, "done": done, "total": total, "note": note[:90], "eta": eta, **({"frac": round(frac, 4)} if frac is not None else {})})

    def finish(self, note=""):
        self._write({"stage": "done", "done": 1, "total": 1, "note": note, "eta": 0, "seconds": round(time.time() - self.t_start)})

    def _write(self, d):
        d["t"] = time.time()
        tmp = self.path.with_suffix(".tmp"); tmp.write_text(json.dumps(d)); os.replace(tmp, self.path)


def read(name, kind):
    try: return json.loads((CFG.runs / name / f"{kind}.progress.json").read_text())
    except (OSError, ValueError): return None

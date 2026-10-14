"""Keep only the best clips of a prepared dataset.

Prepare can produce far more speech than a voice needs. After it has written the transcripts it scores every clip and keeps the best
`max_minutes` of them (at most 270). The rest are removed from the dataset for good, so checking the transcripts, exporting, backing
up and training all work on a small set. Your original recordings are never touched.
"""
import json
from . import voices
from .config import CFG

MAX_MINUTES = 270            # no voice is ever trained on more speech than this; 60 to 120 is the recommended range


def limit(name):
    """Minutes of speech to keep for this voice."""
    return min(MAX_MINUTES, voices.read(name).get("max_minutes") or MAX_MINUTES)


def _dur(root, f):
    p = root / "wavs" / f
    return p.stat().st_size / (2 * 22050) if p.exists() else 0.0


def choose(rows, durs, scores, approved, lim_s):
    """Indices of the clips to keep. Clips with a transcript warning, and clips that sound like another speaker (unless you approved
    them), are left out first. The rest are kept closest-to-your-speaker first until the limit is reached; with no scores, evenly
    spread through the whole dataset. The same clips are chosen every time."""
    def bad(i):
        f, _, t = rows[i].partition("|"); d = durs[i]; w = len(t.split()) / d if d > 0 else 0
        return d <= 0 or len(t.strip()) < 3 or w > 5.5 or (d > 2.5 and w < 1.0) or ((scores.get(f) or {}).get("lv", 0) == 2 and f not in approved)
    idx = list(range(len(rows))); good = [i for i in idx if not bad(i)] or idx
    d_of = lambda i: (scores.get(rows[i].partition("|")[0]) or {}).get("d")
    if any(d_of(i) is not None for i in good): order = sorted(good, key=lambda i: (d_of(i) is None, d_of(i) if d_of(i) is not None else 0, i))
    else: order = sorted(good, key=lambda i: (i * 7919) % len(good))
    keep, tot = set(), 0.0
    for i in order:
        if tot + durs[i] > lim_s: continue
        keep.add(i); tot += durs[i]
    return keep


def read_json(p, default):
    try: return json.loads(p.read_text())
    except (OSError, ValueError): return default


def cull(name, lim=None, log=print, score=True, frac_range=(0.0, 1.0), prog=None):
    """Score the clips (if they have not been) and delete everything beyond the best `lim` minutes. Returns (kept, dropped)."""
    root = CFG.data / name; md = root / "metadata.csv"; lim = lim or limit(name)
    rows = [l for l in md.read_text(encoding="utf-8").splitlines() if "|" in l]
    durs = [_dur(root, r.partition("|")[0]) for r in rows]
    if sum(durs) / 60 <= lim:
        log(f"The dataset has {sum(durs) / 60:.0f} min, within your {lim}-minute limit: nothing to remove."); return len(rows), 0
    sp = CFG.runs / name / "clipscores.json"
    if score and not sp.exists():
        try:
            from . import speakers
            speakers.score_clips(name, log=log, frac_range=frac_range, prog=prog)
        except (SystemExit, Exception) as e:
            log(f"Could not check which clips sound most like your speaker ({e}). Choosing by transcript quality and an even spread instead.")
    scores = read_json(sp, {}).get("scores", {}); ok = set(read_json(CFG.runs / name / "prep.approved.json", []))
    keep = choose(rows, durs, scores, ok, lim * 60)
    kept_f = {rows[i].partition("|")[0] for i in keep}
    for r in rows:
        f = r.partition("|")[0]
        if f not in kept_f:
            (root / "wavs" / f).unlink(missing_ok=True); (root / "wavs.untrimmed" / f).unlink(missing_ok=True)
    def refilter(p):
        if p.exists(): p.write_text("\n".join(l for l in p.read_text(encoding="utf-8").splitlines() if l.partition("|")[0] in kept_f) + "\n", encoding="utf-8")
    refilter(md); refilter(root / "metadata.orig.csv")
    (root / "metadata.train.csv").unlink(missing_ok=True)
    if sp.exists(): sp.write_text(json.dumps({**read_json(sp, {}), "scores": {f: v for f, v in scores.items() if f in kept_f}}))
    ap = CFG.runs / name / "prep.approved.json"
    if ap.exists(): ap.write_text(json.dumps(sorted(set(read_json(ap, [])) & kept_f)))
    log(f"Kept the best {len(keep)} of {len(rows)} clips ({sum(durs[i] for i in keep) / 60:.0f} of {sum(durs) / 60:.0f} min). The others were removed from this voice's dataset; your recordings are untouched.")
    return len(keep), len(rows) - len(keep)

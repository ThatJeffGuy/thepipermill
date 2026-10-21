"""Automatic transcript check: listen to every clip again and compare with the stored text.

A second, independent Whisper pass (plain decoding, not the careful search Prepare used) writes down what it hears in each clip. Where it
agrees with the stored text the clip is confident and is hidden in the Correction step; where it disagrees, the clip is shown with what the
checker heard, as a one-click suggestion. Agreement between two separate passes is a much better sign than a single pass's own confidence.
Result: runs/<voice>/confidence.json  {file: {"heard": text, "wer": share of words that differ}}
"""
import json, os, re
from . import textnorm
from .config import CFG
from .progress import Progress

BATCH = 16
AGREE = 0.05                # up to this share of different words still counts as agreeing


_SAME = (("gonna", "going to"), ("wanna", "want to"), ("gotta", "got to"), ("kinda", "kind of"), ("sorta", "sort of"), ("lemme", "let me"), ("gimme", "give me"), ("dunno", "do not know"))


def _words(t):
    t = (t or "").lower().replace("’", "'")
    for a, b in _SAME: t = re.sub(rf"\b{a}\b", b, t)
    return re.findall(r"[a-z0-9']+", t)


def wer(ref, hyp):
    """Share of the reference words that differ (word-level edit distance / number of reference words)."""
    r, h = _words(ref), _words(hyp)
    if not r: return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1])); prev, d[j] = d[j], cur
    return d[len(h)] / len(r)


def path(name): return CFG.runs / name / "confidence.json"


def read(name):
    try: return json.loads(path(name).read_text())
    except (OSError, ValueError): return None


def run(name, log=print):
    import soundfile as sf, torch
    from transformers import pipeline
    root = CFG.data / name
    rows = [l.rstrip("\n").split("|", 1) for l in open(root / "metadata.csv", encoding="utf-8") if "|" in l]
    if not rows: raise SystemExit("There are no clips to check.")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu": log("WARNING: no GPU found; this will be slow.")
    asr = pipeline("automatic-speech-recognition", model=CFG["whisper_model"], device=dev, torch_dtype=torch.float16 if dev == "cuda" else torch.float32)
    lang = {"en-us": "english", "en-gb": "english", "en": "english"}.get(CFG["espeak_voice"].lower())
    gen = {"task": "transcribe", "num_beams": 1, **({"language": lang} if lang else {})}
    prog = Progress(name, "verify"); old = read(name) or {}; out = {}; done = 0
    size = lambda f: (root / "wavs" / f).stat().st_size
    for f, t in rows:                                       # clips whose audio has not changed since the last check keep their result
        if f in old and old[f].get("size") == size(f): out[f] = old[f]
    todo = [(f, t) for f, t in rows if f not in out]
    if not todo: log("Every clip has already been checked."); prog.finish("nothing new"); return
    log(f"Listening again to {len(todo)} clip(s) (of {len(rows)}) and comparing with the text...")
    rows = todo
    while done < len(rows):
        chunk = rows[done:done + BATCH]; prog.update("Listening to the clips again", done, len(rows), f"{done}/{len(rows)} clips", frac=done / len(rows))
        audio = [{"raw": sf.read(root / "wavs" / f, dtype="float32")[0], "sampling_rate": 22050} for f, _ in chunk]
        try: res = asr(audio, batch_size=len(audio), generate_kwargs=gen)
        except ValueError: res = asr(audio, batch_size=len(audio), generate_kwargs={"num_beams": 1})        # an English-only model takes no language/task options
        for (f, t), r in zip(chunk, res):
            heard = textnorm.normalize(" ".join(r["text"].split()), CFG["espeak_voice"]).replace("|", " ")
            out[f] = {"heard": heard, "size": size(f)}
        done += len(chunk)
        if done % 80 < len(chunk): log(f"  {done}/{len(rows)}")
    tmp = path(name).with_suffix(".tmp"); tmp.write_text(json.dumps(out)); os.replace(tmp, path(name))
    cur = {f: t for f, t in (l.rstrip("\n").split("|", 1) for l in open(root / "metadata.csv", encoding="utf-8") if "|" in l)}
    ok = sum(1 for f, v in out.items() if f in cur and wer(cur[f], v["heard"]) <= AGREE)
    log(f"\nDone: {ok} of {len(cur)} clips agree with the text ({100 * ok // max(1, len(cur))}%); the others will be shown for correction.")
    prog.finish(f"{ok}/{len(cur)} agree")

"""Reprocess after you edit the words.

In "Check the words" you can delete the words another person said from the start or end of a clip's text. This finds when each
remaining word is spoken (Whisper, word by word) and cuts the clip's audio down to just those words, so the voice is only
trained on the person you chose. Clips whose text you only corrected (spelling) are left as they are.

`metadata.orig.csv` is the transcript as it was before your edits; a clip counts as edited when its words differ from it.
The original audio of every trimmed clip is kept in `wavs.untrimmed/`.
"""
import difflib, re, shutil
from pathlib import Path
from . import prep, textnorm
from .config import CFG
from .progress import Progress

PAD, MIN_S, EDGE = 0.08, 1.0, 0.35      # seconds kept around the kept words / shortest clip worth keeping / edge slack: closer than this = nothing to cut


def _tok(s): return re.findall(r"[a-z0-9']+", (s or "").lower())


def _read(path):
    rows = {}
    if path.exists():
        for line in open(path, encoding="utf-8"):
            f, _, t = line.rstrip("\n").partition("|")
            if f: rows[f] = t
    return rows


def pending(name):
    """Clips whose words you changed since the transcript was written."""
    root = CFG.data / name; cur, orig = _read(root / "metadata.csv"), _read(root / "metadata.orig.csv")
    return [f for f, t in cur.items() if f in orig and _tok(t) != _tok(orig[f])]


def span(words, new_text, lang="en-us"):
    """(start, end, status) of the kept words inside a clip. `words` = [(text, start, end)] from Whisper.
    status: ok | added (you only ADDED words Whisper did not hear: the text is kept, the audio is not cut) | nomatch (the words cannot be found in the audio) | middle (words were removed from the middle: cannot be cut)."""
    toks = []                                    # (token, index of the Whisper word it came from); numbers spoken as words line up
    for i, (w, _, _) in enumerate(words):
        toks += [(t, i) for t in _tok(textnorm.normalize(w, lang))]
    want = _tok(new_text)
    if not toks or not want: return None, None, "nomatch"
    blocks = [b for b in difflib.SequenceMatcher(None, [t for t, _ in toks], want, autojunk=False).get_matching_blocks() if b.size]
    hit = sum(b.size for b in blocks)
    if hit >= 0.9 * len(toks) and len(want) > len(toks): return None, None, "added"       # everything Whisper heard is in your text, and you added more: you heard words it missed
    if hit < 0.6 * len(want): return None, None, "nomatch"
    a, z = blocks[0].a, blocks[-1].a + blocks[-1].size
    if (z - a) - sum(b.size for b in blocks) > 2: return None, None, "middle"       # the audio between first and last match has extra words
    return words[toks[a][1]][1], words[toks[z - 1][1]][2], "ok"


def run(name, log=print):
    import soundfile as sf, torch
    from transformers import pipeline
    root = CFG.data / name; wavs = root / "wavs"; md, base = root / "metadata.csv", root / "metadata.orig.csv"
    todo = pending(name)
    if not todo: log("Nothing to reprocess: no clip's words were changed."); Progress(name, "trim").finish("nothing to do"); return
    cur, orig = _read(md), _read(base)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    asr = pipeline("automatic-speech-recognition", model=CFG["whisper_model"], device=dev, torch_dtype=torch.float16 if dev == "cuda" else torch.float32)
    lang = {"en-us": "english", "en-gb": "english", "en": "english"}.get(CFG["espeak_voice"].lower())
    gen = {"task": "transcribe", **({"language": lang} if lang else {})}
    keep_dir = root / "wavs.untrimmed"; keep_dir.mkdir(exist_ok=True)
    prog = Progress(name, "trim"); trimmed = same = bad = 0; problems = []
    for i, f in enumerate(todo, 1):
        prog.update("Cutting clips to your edited words", i - 1, len(todo), f)
        p = wavs / f
        if not p.exists(): continue
        y, sr = sf.read(p, dtype="float32"); dur = len(y) / sr
        try:
            res = asr({"raw": y, "sampling_rate": sr}, return_timestamps="word", generate_kwargs=gen)
            words = [(c["text"].strip(), c["timestamp"][0] or 0.0, c["timestamp"][1] if c["timestamp"][1] is not None else dur) for c in res.get("chunks", [])]
        except Exception as e:
            problems.append(f"{f}: could not find the words in the audio ({str(e)[:60]})"); bad += 1; continue
        s, e, st = span(words, cur[f], CFG["espeak_voice"])
        if st == "added": same += 1; orig[f] = cur[f]; log(f"  {f}: words added that the speech recogniser missed: text updated, audio left as it was"); continue
        if st == "nomatch": problems.append(f"{f}: your words do not match what is said, so the audio was left alone. Remove the clip or restore the words."); bad += 1; continue
        if st == "middle": problems.append(f"{f}: words were removed from the middle, which cannot be cut. Remove the clip, or remove words only from the start or end."); bad += 1; continue
        if s <= EDGE and e >= dur - EDGE:
            same += 1; orig[f] = cur[f]; continue                                  # a spelling fix: nothing to cut
        a, b = max(0.0, s - PAD), min(dur, e + PAD)
        if b - a < MIN_S:
            problems.append(f"{f}: only {b - a:.1f} s would be left, too short to use. Remove the clip instead."); bad += 1; continue
        if not (keep_dir / f).exists(): shutil.copy2(p, keep_dir / f)
        sf.write(p, prep.level(y[int(a * sr):int(b * sr)]), sr, subtype="PCM_16")
        orig[f] = cur[f]; trimmed += 1
        log(f"  {f}: {dur:.1f} s -> {b - a:.1f} s")
    base.write_text("".join(f"{f}|{t}\n" for f, t in orig.items()), encoding="utf-8")
    shutil.rmtree(CFG.cache / name, ignore_errors=True)                            # Piper's cache is keyed by clip+text
    if trimmed: (CFG.runs / name / "clipscores.json").unlink(missing_ok=True)      # cut clips sound different: score them again
    log(f"\nDone: {trimmed} clip(s) cut to your words, {same} spelling-only edit(s) kept as they were, {bad} need your attention.")
    for m in problems[:30]: log(f"  - {m}")
    if trimmed: log("Listen to a few trimmed clips (press ▶) to be sure they sound right.")
    prog.finish(f"{trimmed} clips trimmed")

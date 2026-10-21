"""Reprocess after you edit the words.

In the transcript check you can delete words anywhere in a clip's text: the start, the end, or the middle, as many stretches as you like (for example
the other person's lines in a back-and-forth conversation). This finds when each word is spoken (Whisper, word by word) and keeps only the
stretches whose words you kept. One kept stretch trims the clip; several kept stretches split it into separate clips (never joined together,
so there is no unnatural splice). Pieces shorter than a second are dropped. Clips whose text you only corrected (spelling) are left as they are.

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


def ndel_hint(orig_text, new_text):
    """True when the edit removed any words (cheap check used only to decide whether a timing warning matters)."""
    return len(_tok(new_text)) < len(_tok(orig_text))


def _align(xs, ys, floor=0.5):
    """Pair up the words of two short lists by how alike they are, keeping their order (dynamic programming). -> [(index in xs, index in ys)]"""
    n, m = len(xs), len(ys)
    sc = [[difflib.SequenceMatcher(None, x, y).ratio() for y in ys] for x in xs]
    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            take = sc[i][j] + best[i + 1][j + 1] if sc[i][j] >= floor else -1
            best[i][j] = max(take, best[i + 1][j], best[i][j + 1])
    out, i, j = [], 0, 0
    while i < n and j < m:
        if sc[i][j] >= floor and abs(best[i][j] - (sc[i][j] + best[i + 1][j + 1])) < 1e-9: out.append((i, j)); i += 1; j += 1
        elif best[i + 1][j] >= best[i][j + 1]: i += 1
        else: j += 1
    return out


def fade(clip, sr, start, end, ms=8):
    """A very short fade on each edge that was cut, so a cut through a sound never leaves a click."""
    import numpy as np
    n = min(int(sr * ms / 1000), len(clip) // 4); y = clip.copy()
    if start and n: y[:n] *= np.linspace(0.0, 1.0, n, dtype="float32")
    if end and n: y[-n:] *= np.linspace(1.0, 0.0, n, dtype="float32")
    return y


def _tok_sp(s):
    """[(token, char start, char end)] for the words of a text."""
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[a-z0-9']+", (s or "").lower())]


def plan(orig_text, new_text, words, dur, lang="en-us"):
    """Work out which stretches of a clip to keep after the user deleted words from its text. Pure function (no audio, no disk).

    orig_text = the transcript before the edit, new_text = after, words = [(text, start, end)] from Whisper for this clip, dur = clip seconds.
    Returns (status, runs, deleted). status: nocut (nothing was deleted: a spelling fix or added words) | nomatch (the text does not fit the audio)
    | removed (every word was deleted) | ok. runs = [(start, end, text)]: one per stretch of kept words, in order; any number of deleted stretches
    anywhere in the clip (start, middle, end, several times) simply split it into separate pieces. deleted = number of deleted words."""
    ot, et = _tok(orig_text), _tok_sp(new_text)
    if not et: return "removed", [], len(ot)
    wt = []                                               # (token, index of the Whisper word it came from); numbers spoken as words line up
    for i, (w, _, _) in enumerate(words):
        wt += [(t, i) for t in _tok(textnorm.normalize(w, lang))]
    if not ot or not wt: return "nomatch", [], 0
    sm = difflib.SequenceMatcher(None, ot, [t for t, _, _ in et], autojunk=False)
    keep, attach = [True] * len(ot), {}                   # attach[i] = edited-text token indexes that belong to original token i
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1): attach.setdefault(i1 + k, []).append(j1 + k)
        elif tag == "replace":
            if i2 - i1 == j2 - j1:                        # same number of words: spelling fixes, kept one for one
                for k in range(i2 - i1): attach.setdefault(i1 + k, []).append(j1 + k)
            else:                                         # a fix next to a deletion: match the words that look alike, the rest of the old words were deleted
                pairs = _align(ot[i1:i2], [t for t, _, _ in et[j1:j2]]); got_i = set()
                for ai, aj in pairs: attach.setdefault(i1 + ai, []).append(j1 + aj); got_i.add(ai)
                for ai in range(i2 - i1):
                    if ai not in got_i: keep[i1 + ai] = False
                left = [j for j in range(j1, j2) if j not in {j1 + aj for _, aj in pairs}]
                if left: attach.setdefault(max(i1, (i1 + max(got_i)) if got_i else i1), []).extend(left)
        elif tag == "delete":
            for i in range(i1, i2): keep[i] = False
        elif tag == "insert":                             # added words ride with the token before them (or after, at the very start)
            attach.setdefault(max(0, i1 - 1) if i1 else 0, []).extend(range(j1, j2))
    deleted = keep.count(False)
    if not deleted: return "nocut", [], 0
    # where each original token is spoken: match the original text to Whisper's words
    wm = difflib.SequenceMatcher(None, ot, [t for t, _ in wt], autojunk=False); at = {}
    for b in wm.get_matching_blocks():
        for k in range(b.size): at[b.a + k] = words[wt[b.b + k][1]][1:3]
    if len(at) < 0.6 * len(ot): return "nomatch", [], deleted
    runs, cur = [], []
    for i in range(len(ot) + 1):
        if i < len(ot) and keep[i]: cur.append(i); continue
        if cur:
            timed = [at[k] for k in cur if k in at]
            if timed:
                s0, e0 = min(t[0] for t in timed), max(t[1] for t in timed)
                prev = next((at[k] for k in range(cur[0] - 1, -1, -1) if k in at), None) if cur[0] > 0 else None       # the closest word before the run (deleted, since runs break at deletions)
                nxt = next((at[k] for k in range(cur[-1] + 1, len(ot)) if k in at), None) if cur[-1] < len(ot) - 1 else None
                lo = max(prev[1] + (s0 - prev[1]) / 2 if prev and cur[0] > 0 and not keep[cur[0] - 1] else 0.0, s0 - PAD)
                hi = min(nxt[0] - (nxt[0] - e0) / 2 if nxt and cur[-1] < len(ot) - 1 and not keep[cur[-1] + 1] else dur, e0 + PAD)
                js = sorted({j for k in cur for j in attach.get(k, [])})
                if js:
                    a, z = et[js[0]][1], et[js[-1]][2]
                    while z < len(new_text) and new_text[z] in ".,!?…\"”’)": z += 1             # keep the punctuation that ends the words
                    runs.append((max(0.0, lo), min(dur, hi), new_text[a:z].strip()))
        cur = []
    return "ok", runs, deleted


def run(name, log=print):
    import soundfile as sf, torch
    from transformers import pipeline
    root = CFG.data / name; wavs = root / "wavs"; md, base = root / "metadata.csv", root / "metadata.orig.csv"
    todo = set(pending(name))
    if not todo: log("Nothing to reprocess: no clip's words were changed."); Progress(name, "trim").finish("nothing to do"); return
    cur, orig = _read(md), _read(base)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    asr = pipeline("automatic-speech-recognition", model=CFG["whisper_model"], device=dev, torch_dtype=torch.float16 if dev == "cuda" else torch.float32)
    lang = {"en-us": "english", "en-gb": "english", "en": "english"}.get(CFG["espeak_voice"].lower())
    gen = {"task": "transcribe", **({"language": lang} if lang else {})}
    keep_dir = root / "wavs.untrimmed"; keep_dir.mkdir(exist_ok=True)
    prog = Progress(name, "trim"); cut = split = removed = same = bad = 0; problems = []; pieces_of = {}; pend = set(); changed = set()      # pieces_of: file -> [(file, text)] it becomes. pend: clips that could not be changed and stay marked as edited
    todo_list = [f for f in cur if f in todo]
    for i, f in enumerate(todo_list, 1):
        prog.update("Cutting clips to your edited words", i - 1, len(todo_list), f)
        p = wavs / f
        if not p.exists(): pieces_of[f] = [(f, cur[f])]; pend.add(f); continue
        y, sr = sf.read(p, dtype="float32"); dur = len(y) / sr
        try:
            try: res = asr({"raw": y, "sampling_rate": sr}, return_timestamps="word", generate_kwargs=gen)
            except ValueError: res = asr({"raw": y, "sampling_rate": sr}, return_timestamps="word")      # an English-only model takes no language/task options
            words = [(c["text"].strip(), c["timestamp"][0] or 0.0, c["timestamp"][1] if c["timestamp"][1] is not None else dur) for c in res.get("chunks", [])]
        except Exception as e:
            problems.append(f"{f}: could not find the words in the audio ({str(e)[:60]})"); bad += 1; pieces_of[f] = [(f, cur[f])]; pend.add(f); continue
        long_w = [w for w, a, b in words[:-1] if b - a > 1.2]
        if long_w and ndel_hint(orig.get(f, ""), cur[f]): problems.append(f"{f}: the timing of the word \"{long_w[0]}\" looks wrong (it lasts over a second), so there may be speech the text does not include. Listen to this clip after the cut.")
        st, runs, ndel = plan(orig.get(f, ""), cur[f], words, dur, CFG["espeak_voice"])
        def untouched(msg=None):
            nonlocal bad
            if msg: problems.append(msg); bad += 1
            pieces_of[f] = [(f, cur[f])]; pend.add(f)
        if st == "nocut": same += 1; pieces_of[f] = [(f, cur[f])]; continue                      # a spelling fix, or words added that the recogniser missed: text updated, audio kept
        if st == "nomatch": untouched(f"{f}: the words do not match what is said, so the audio was left alone. Remove the clip or restore the words."); continue
        if not (keep_dir / f).exists(): shutil.copy2(p, keep_dir / f)
        if st == "removed":
            p.unlink(missing_ok=True); removed += 1; changed.add(f); pieces_of[f] = []; log(f"  {f}: all the words were deleted: clip removed"); continue
        good = [(a, b, t) for a, b, t in runs if b - a >= MIN_S]
        for a, b, t in runs:
            if b - a < MIN_S: problems.append(f"{f}: a piece of only {b - a:.1f} s (\"{t[:30]}\") was too short to keep and was dropped.")
        if not good:
            p.write_bytes((keep_dir / f).read_bytes()); untouched(f"{f}: every remaining piece was too short to use. Remove the clip instead."); continue
        stem = f[:-4] if f.endswith(".wav") else f; pieces_of[f] = []; changed.add(f)
        for k, (a, b, t) in enumerate(good, 1):
            fn = f if len(good) == 1 else f"{stem}_{k}.wav"
            sf.write(wavs / fn, fade(prep.level(y[int(a * sr):int(b * sr)]), sr, a > 0.01, b < dur - 0.01), sr, subtype="PCM_16"); pieces_of[f].append((fn, t))
        if len(good) > 1: p.unlink(missing_ok=True); split += 1; log(f"  {f}: {dur:.1f} s -> {len(good)} clips (" + ", ".join(f"{b - a:.1f} s" for a, b, _ in good) + ")")
        else: cut += 1; log(f"  {f}: {dur:.1f} s -> {good[0][1] - good[0][0]:.1f} s")
    # rewrite both transcripts in the original order, with each split clip in place of the clip it came from
    new_cur, new_orig = [], []
    for f in cur:
        if f in pieces_of:
            new_cur += pieces_of[f]; new_orig += [(f, orig.get(f, cur[f]))] if f in pend else pieces_of[f]      # edited clips: what they became (the transcript now matches the audio)
        else: new_cur.append((f, cur[f])); new_orig.append((f, orig.get(f, cur[f])))
    md.write_text("".join(f"{f}|{t}\n" for f, t in new_cur), encoding="utf-8"); base.write_text("".join(f"{f}|{t}\n" for f, t in new_orig), encoding="utf-8")
    shutil.rmtree(CFG.cache / name, ignore_errors=True)                            # Piper's cache is keyed by clip+text
    try:                                                                           # the automatic check no longer describes clips whose audio changed: they will be checked again
        import json
        cp = CFG.runs / name / "confidence.json"
        if cp.exists():
            cf = json.loads(cp.read_text())
            for f in changed: cf.pop(f, None)
            cp.write_text(json.dumps(cf))
    except (OSError, ValueError): pass
    if cut or split or removed: (CFG.runs / name / "clipscores.json").unlink(missing_ok=True)      # cut clips sound different: score them again
    log(f"\nDone: {cut} clip(s) trimmed, {split} clip(s) split into several, {removed} removed, {same} spelling-only edit(s) kept as they were, {bad} need your attention.")
    for m in problems[:30]: log(f"  - {m}")
    if cut or split: log("Listen to a few of the new clips (press ▶) to be sure they sound right.")
    prog.finish(f"{cut + split} clips changed")

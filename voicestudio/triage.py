"""Sorts the prepared clips into what to check first: likely errors, then questionable ones, then a random spot-check.

Pure functions (no disk access) so the order is easy to reason about and test. Tiers:
  1 = problem       something is almost certainly wrong (words don't fit the audio, another speaker, junk text)
  2 = questionable  worth a listen (borderline pace or speaker match, odd characters, cut-off sentence); a random sample if there are many
  3 = spot check    a fixed random sample of the clips nothing suspicious was found in
"""
import hashlib, re

SPOT = 100                       # how many "nothing looks wrong" clips to listen to as a sanity check
MAYBE = 100                      # most "questionable" clips to ask for: a random sample of them, so the picks span all the recordings
_JUNK = re.compile(r"thanks? for watching|subtitles? by|subscribe|like and subscribe|www\.|https?:|\.com\b|♪|\[(music|applause|laughter)|\(music|amara\.org|translated by", re.I)
_WORD = re.compile(r"[A-Za-z']+")


def reasons(r):
    """-> (severity, [reason, ...]) for one row {file, text, dur, warn, score}. Severity >= 10 is a problem, > 0 is questionable."""
    t, dur, sc = (r.get("text") or "").strip(), r.get("dur") or 0, r.get("score")
    words = _WORD.findall(t); n = len(t.split()); wps = n / dur if dur > 0 else 0
    sev, why = 0, []
    def add(s, msg): nonlocal sev; sev += s; why.append(msg)
    if r.get("warn"): add(10, r["warn"])
    if sc and sc.get("lv") == 2: add(10, "Sounds very different from the rest: maybe another speaker or noise")
    elif sc and sc.get("lv") == 1: add(4, "Sounds a bit different from the rest: could be another speaker or noise")
    if not words: add(10, "The text has no real words")
    if _JUNK.search(t): add(10, "The text looks like subtitle or video junk (credits, music tags, web addresses), not speech")
    if re.search(r"\b(\w+)(\W+\1\b){3,}", t, re.I): add(10, "The same word is repeated many times, a typical sign of a transcription glitch")
    if len(re.findall(r"[^\x00-\x7F’‘“”—–…é]", t)) > 2: add(6, "Unusual characters in the text")
    if not r.get("warn"):
        if wps > 4.8: add(5, "Fast for this length: the text may have extra words")
        elif dur > 2.5 and wps < 1.6: add(5, "Slow for this length: words may be missing from the text")
    if dur and dur < 1.5: add(3, "Very short clip")
    if dur > 14: add(3, "Long clip: easy for the text to drift from the audio")
    if n and n < 2 and not r.get("warn"): add(4, "Only one word")
    if re.search(r"[\[\]\(\)\*#_<>{}|]", t): add(4, "Brackets or symbols in the text, which the voice would try to read out")
    if re.search(r"\d", t): add(3, "Contains digits: check they are written the way they are spoken")
    if t and t[-1] not in ".?!…\"”'’": add(2, "Doesn't end like a sentence: the clip may be cut off")
    if t and t.upper() == t and len(words) > 2: add(3, "All capitals: may be a caption, not speech")
    return sev, why


def triage(rows, spot=SPOT, maybe=MAYBE):
    """Adds r['tier'] (1/2/3/0), r['why'] (reasons) and r['rnd'] (stable shuffle key) to every row; returns the counts.
    Tiers 2 and 3 are stable random samples (hash of the file name), so they span the whole recording set and
    don't reshuffle every time the page is opened. Questionable clips beyond the sample go back into the spot-check pool."""
    for r in rows:
        sev, why = reasons(r); r["sev"] = sev; r["why"] = why
        r["rnd"] = hashlib.md5(r["file"].encode()).hexdigest()
        r["tier"] = 1 if sev >= 10 else 2 if sev >= 4 else 0
    q = sorted((r for r in rows if r["tier"] == 2), key=lambda r: r["rnd"])
    for r in q[maybe:]: r["tier"] = 0
    pool = sorted((r for r in rows if r["tier"] == 0), key=lambda r: r["rnd"])
    for r in pool[:spot]:
        r["tier"] = 3; r["why"] = r["why"] or ["Random spot-check: if these are fine, the ones nobody looked at are very likely fine too"]
    c = {1: 0, 2: 0, 3: 0}
    for r in rows:
        if r["tier"]: c[r["tier"]] += 1
    return {"problem": c[1], "questionable": c[2], "spot": c[3], "total": len(rows)}

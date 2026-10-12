"""Make transcripts match how the words were spoken.

Piper reads text with espeak-ng, so what is written must line up with what is said: "15%" -> "fifteen percent",
"$3.50" -> "three dollars and fifty cents", "Dr." -> "doctor", smart quotes and symbols tidied up. English rules only
(other languages get the basic clean-up).
"""
import re, unicodedata

ABBR = {"mr": "mister", "mrs": "missus", "ms": "miss", "dr": "doctor", "prof": "professor", "st": "saint", "mt": "mount", "vs": "versus"}
PUNCT_FIX = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": ", ", "—": ", ", "…": "...", " ": " "}
YEAR_CONTEXT = re.compile(r"(?:\b(?:in|since|of|by|until|till|from|around|circa|year|during|before|after)\s+|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:\d{1,2},?\s+)?)$", re.I)


def _words(n, **kw):
    from num2words import num2words
    return num2words(n, **kw)


def _digits(s): return " ".join(_words(int(c)) for c in s)


def _clock(m):
    h, mi = int(m.group(1)), int(m.group(2))
    return f"{_words(h)} " + ("o'clock" if mi == 0 else f"oh {_words(mi)}" if mi < 10 else _words(mi))


def _money(m):
    whole = int(m.group(1).replace(",", "")); cents = m.group(2)
    s = f"{_words(whole)} dollar{'' if whole == 1 else 's'}"
    if cents and int(cents): s += f" and {_words(int(cents))} cent{'' if int(cents) == 1 else 's'}"
    return s


def normalize(text, lang="en-us"):
    t = unicodedata.normalize("NFKC", text)
    for a, b in PUNCT_FIX.items(): t = t.replace(a, b)
    t = re.sub(r"\[[^\]]*\]|\*[^*]*\*", " ", t)                                   # [Music], *laughs*
    t = re.sub(r"[^\x00-\x7fÀ-ɏ]", " ", t) if lang.lower().startswith("en") else t   # emoji, symbols, other scripts
    if lang.lower().startswith("en"):
        t = re.sub(r"\$(\d[\d,]*)(?:\.(\d\d))?", _money, t)
        t = re.sub(r"(\d[\d,]*(?:\.\d+)?)\s*%", lambda m: m.group(1) + " percent", t)
        t = re.sub(r"\b(\d{1,2}):(\d\d)\b", _clock, t)
        t = re.sub(r"\b([ap])\.?m\.?\b", lambda m: m.group(1).lower() + " m", t, flags=re.I)
        t = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: _words(int(m.group(1)), to="ordinal"), t, flags=re.I)
        def dotted_time(m):               # Whisper often writes "8.30" for half past eight; after "at/by/until..." that is a time
            h, mi = int(m.group(1)), int(m.group(2))
            if re.match(r"\s*(?:percent|dollar|cent|km|kilomet|met[er]|mile|kg|kilo|pound|lit|degree|times|x\b)", t[m.end():m.end() + 14], re.I): return m.group(0)
            if h <= 24 and mi < 60 and re.search(r"\b(?:at|by|until|till|from|around|before|after|since|between|to)\s*$", t[:m.start()], re.I):
                return _clock(m)
            return m.group(0)
        t = re.sub(r"\b(\d{1,2})\.(\d\d)\b", dotted_time, t)
        t = re.sub(r"\b(\d+)\.(\d+)\b", lambda m: f"{_words(int(m.group(1)))} point {_digits(m.group(2))}", t)
        t = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", lambda m: m.group(0).replace(",", ""), t)

        def num(m):
            n, before = int(m.group(0)), t[:m.start()]
            if 1100 <= n <= 2099 and YEAR_CONTEXT.search(before): return _words(n, to="year")
            return _words(n)
        t = re.sub(r"(?<![\w.])\d+(?![\w])", num, t)
        for a, b in (("&", " and "), ("+", " plus "), ("@", " at "), ("#", " number "), ("=", " equals "), ("°", " degrees "), ("~", " "), ("/", " ")):
            t = t.replace(a, b)
        t = re.sub(r"\b(" + "|".join(ABBR) + r")\.", lambda m: ABBR[m.group(1).lower()], t, flags=re.I)
        t = re.sub(r"[^A-Za-zÀ-ɏ0-9 .,!?;:'\"-]", " ", t)                    # anything else left over
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([.,!?;:])", r"\1", t)
    if t and t[-1] not in ".!?": t += "."
    return t


def looks_like_loop(t):
    """Whisper sometimes repeats a phrase over and over on noise or mumbling."""
    s = t.lower()
    if re.search(r"(\b[\w' ]{3,40}?\b)(?:[ ,.!?]+\1){2,}", s): return True
    w = s.split()
    return len(w) >= 8 and len(set(w)) / len(w) < 0.35

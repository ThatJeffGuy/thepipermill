"""Hide known, harmless noise from the console the page shows (the full log stays on disk).

Third-party libraries print warnings that mean nothing to someone training a voice: library deprecations, tips, notices about
optional features. Real problems (a Traceback, an Error, an out-of-memory, a skipped file) are never hidden.
"""
import re

# a Python warning header: "/path/file.py:103: UserWarning: ..." (its message and source lines follow it)
_HEADER = re.compile(r"^\S+\.py:\d+: \w*Warning\b")
# lines that belong to a warning block but are not indented
_CONTINUES = re.compile(r"^(It can be re-enabled|See https?://|To (disable|silence|suppress)|If you (want|wish) to)", re.I)
_QUIET = [re.compile(p, re.I) for p in (
    r"^warnings\.warn\(",
    r"^Fetching \d+ files",
    r"^>>> ",
    r"unauthenticated requests to the HF Hub",
    r"^(GPU|TPU|HPU|IPU) available:",
    r"Tip: For seamless cloud logging|litlogger",
    r"^(Using|Trainer\(|You are using a CUDA device .* Tensor Cores|LOCAL_RANK|Lightning automatically upgraded)",
    r"Model was trained with (pyannote\.audio|torch) ",
    r"Due to a bug fix in https://github\.com/huggingface/transformers",
    r"attention mask is not set|`return_token_timestamps` is deprecated|Passing a tuple of `past_key_values`|forced_decoder_ids|pad_token_id.* eos_token_id|Special tokens have been added|slow image processor",
    r"^\W*(FutureWarning|DeprecationWarning|UserWarning|ReproducibilityWarning)\b",
)]
_REAL = re.compile(r"Traceback|Error|Exception|SystemExit|SKIPPED|Killed|out of memory|\bOOM\b|did not finish|failed", re.I)


def clean(text):
    """The lines of `text` worth showing."""
    out, skipping = [], False
    for line in re.split(r"[\r\n]+", text):
        s = line.strip()
        if not s: continue
        if _HEADER.match(s): skipping = True; continue
        if skipping and (line[:1] in " \t" or _CONTINUES.match(s)): continue         # the rest of that warning
        skipping = False
        if _REAL.search(s) and "Warning" not in s: out.append(s); continue            # never hide a real problem
        if any(q.search(s) for q in _QUIET): continue
        out.append(s)
    return out

"""Settings. Everything is optional: defaults work out of the box.

Config file (TOML), first found wins:
  $VOICESTUDIO_CONFIG, then ~/.config/voice-studio/config.toml
See config.example.toml for every option.
"""
import copy, os, re, tomllib
from pathlib import Path

DEFAULTS = {
    "home": "~/voice-studio-data",          # where recordings, datasets, checkpoints and finished voices live
    "host": "127.0.0.1",                     # keep this on localhost unless you know what you are doing
    "port": 8765,
    "allowed_hosts": ["127.0.0.1", "localhost"],   # Host headers the web page accepts
    "espeak_voice": "en-us",                 # language of your recordings (espeak-ng voice name)
    "audio_language": "",                    # for videos with several audio tracks: prefer this language, e.g. "eng"
    "whisper_model": "openai/whisper-large-v3-turbo",   # used to write down what you said
    "train_precision": "",                   # "" = Piper default (full precision). Try "bf16-mixed" for a faster run; compare the result by ear
    "whisper_beams": 5,                      # Prepare's transcription search width: 1 is much faster, 5 is a little more accurate
    "batch_size": 0,                         # 0 = choose from your GPU memory
    "hf_token": "",                          # Hugging Face read token (only for the speaker-finding model, which is gated)
    "diarization_model": "pyannote/speaker-diarization-community-1",
    "speaker_match_distance": 0.55,          # lower = stricter about calling two voices the same person
    "protected_names": [],                   # voice names that can never be created
    "notify_url": "",                        # optional: POST here when a job finishes (a Home Assistant webhook, ntfy topic, ...)
    "notify_format": "json",                 # json = {"title","message","voice","job","status"} | text = just the message (ntfy)
}

NAME_RX = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def _merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict): _merge(a[k], v)
        else: a[k] = v
    return a


class Config:
    def __init__(self):
        self.path = Path(os.environ.get("VOICESTUDIO_CONFIG") or "~/.config/voice-studio/config.toml").expanduser()
        d = copy.deepcopy(DEFAULTS)
        if self.path.exists():
            with open(self.path, "rb") as f: _merge(d, tomllib.load(f))
        self.d = d
        self.home = Path(os.path.expanduser(d["home"])).resolve()
        for n in ("data", "runs", "out", "base", "voices", "cache"):
            setattr(self, n, self.home / n)

    def __getitem__(self, k): return self.d[k]

    def ensure_dirs(self):
        for n in ("data", "runs", "out", "base", "voices", "cache"): getattr(self, n).mkdir(parents=True, exist_ok=True)

    def check_name(self, n):
        n = (n or "").strip()
        if not NAME_RX.match(n): raise ValueError("Voice name: 1-40 characters: letters, digits, - and _ only. Capital letters count (Joe and joe are different names).")
        if n in self.d["protected_names"]: raise ValueError(f"'{n}' is a protected name.")
        return n


CFG = Config()

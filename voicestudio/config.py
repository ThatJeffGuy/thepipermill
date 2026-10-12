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
    "whisper_model": "openai/whisper-large-v3-turbo",   # used to write down what you said
    "batch_size": 0,                         # 0 = choose from your GPU memory
    "max_upload_gb": 4,
    "protected_names": [],                   # voice names that can never be created or installed over
    "install": {                             # what the "Install" button does (download links always work)
        "mode": "none",                      # none | local | ssh | script
        "local_dir": "",                     # local: copy .onnx/.json here
        "ssh_host": "", "ssh_port": 22,      # ssh: scp to  ssh_host:remote_dir
        "remote_dir": "",
        "post_command": "",                  # local/ssh: run afterwards (e.g. restart your Piper service)
        "script": "",                        # script: run  script <name> <onnx> <json>
    },
}

NAME_RX = re.compile(r"^[a-z0-9_-]{1,40}$")


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
        for n in ("inbox", "data", "runs", "out", "base", "voices", "cache"):
            setattr(self, n, self.home / n)

    def __getitem__(self, k): return self.d[k]

    def ensure_dirs(self):
        for n in ("inbox", "data", "runs", "out", "base", "voices", "cache"): getattr(self, n).mkdir(parents=True, exist_ok=True)

    def check_name(self, n):
        n = (n or "").strip()
        if not NAME_RX.match(n): raise ValueError("Voice name: 1-40 characters, lowercase letters, digits, - and _ only.")
        if n in self.d["protected_names"]: raise ValueError(f"'{n}' is a protected name.")
        return n


CFG = Config()

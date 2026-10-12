"""The web page and its small JSON API. Standard library only. Localhost by default.

Long jobs (prepare, train, export) run as separate processes that keep going if you close the page.
Only one job runs at a time because they share the GPU.
"""
import json, os, re, shutil, signal, subprocess, sys, tempfile, threading, zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote
from . import base, install, progress, speakers, train as train_mod, trim, voices
from .config import CFG

WEB = Path(__file__).parent / "web"
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".mp4", ".mkv", ".webm"}
JOBS = {}   # voice -> {"kind", "proc", "log"}


def alive(pid):
    try:
        os.kill(pid, 0)
        with open(f"/proc/{pid}/stat") as f:        # a finished child nobody waited for is a zombie: not alive
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def train_pid(n):
    try:
        pid = int((CFG.runs / n / "train.pid").read_text().strip()); return pid if alive(pid) else None
    except (OSError, ValueError):
        return None


JOB_KINDS = ("speakers", "match", "prep", "trim", "export")


def job_pids(n):
    """Running one-off jobs (not training) of a voice: [(kind, pid)]. Their pids are on disk so this survives a page-server restart."""
    out = []
    for k in JOB_KINDS:
        try:
            pid = int((CFG.runs / n / f"{k}.pid").read_text().strip())
            if alive(pid): out.append((k, pid))
        except (OSError, ValueError): pass
    return out


def busy():
    if CFG.runs.exists():
        for pf in CFG.runs.glob("*/train.pid"):
            if train_pid(pf.parent.name): return pf.parent.name, "training"
        for d in CFG.runs.iterdir():
            j = job_pids(d.name) if d.is_dir() else []
            if j: return d.name, j[0][0]
    for n, j in JOBS.items():
        if j["proc"].poll() is None: return n, j["kind"]
    return None


def start_job(n, kind, args, meta=None):
    b = busy()
    if b: raise RuntimeError(f"'{b[0]}' is busy ({b[1]}). Only one job at a time: they share the GPU.")
    d = CFG.runs / n; d.mkdir(parents=True, exist_ok=True)
    log = d / f"{kind}.log"
    p = subprocess.Popen([sys.executable, "-u", "-m", "voicestudio", *args], stdout=open(log, "wb"), stderr=subprocess.STDOUT,
                         start_new_session=True)
    threading.Thread(target=p.wait, daemon=True).start()           # reap it when it ends
    if kind == "train":
        (d / "train.pid").write_text(str(p.pid)); (d / "train.meta").write_text(json.dumps(meta or {}))
    else:
        JOBS[n] = {"kind": kind, "proc": p, "log": log}
        (d / f"{kind}.pid").write_text(str(p.pid))


def start_over(n, training=False):
    """Back to the beginning for one voice. Keeps: the recordings folder (never touched), the name, male/female, the decoded-audio cache.
    Clears: the speaker scan and your choices, ticked clips, time window, prepared clips and edits. Training only if asked."""
    runs = CFG.runs / n
    if runs.exists():
        for c in list(runs.iterdir()):
            if c.name == "diar": continue
            if training or c.name.startswith(("speakers", "match", "prep", "trim", "export")):
                shutil.rmtree(c, ignore_errors=True) if c.is_dir() else c.unlink(missing_ok=True)
    shutil.rmtree(CFG.data / n, ignore_errors=True); shutil.rmtree(CFG.cache / n, ignore_errors=True)
    if training:
        for g in (CFG.out.glob(f"{n}.*"), CFG.out.glob(f"{n}-*")):
            for p in g: p.unlink(missing_ok=True)
    JOBS.pop(n, None)
    kw = dict(keep_people=[], keep_from=None, keep_until=None, ref_yes=[], ref_no=[], use_match=False, people_names={})
    if training: kw["base_used"] = None
    voices.write(n, **kw)


def tail(path, n=2500):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); f.seek(max(0, f.tell() - n)); t = f.read().decode(errors="ignore")
    except OSError:
        return ""
    ls = [l.strip() for l in re.split(r"[\r\n]+", t) if l.strip() and "arn" not in l[:12]]
    return "\n".join(ls[-6:])


def info(n):
    m = voices.read(n); src = voices.source_dir(n)
    files = speakers.audio_files(n)
    md = CFG.data / n / "metadata.csv"
    clips = sum(1 for _ in open(md, encoding="utf-8")) if md.exists() else 0
    runs = CFG.runs / n
    ck = list(runs.glob("lightning_logs/version_*/checkpoints/*.ckpt")) if runs.exists() else []
    tl = tail(runs / "train.log", 6000) if (runs / "train.log").exists() else ""
    ep = re.findall(r"Epoch (\d+)/(\d+)", tl); mos = re.findall(r"val_mos[:=]\s*([\d.]+)", tl)
    try: total = json.loads((runs / "train.meta").read_text())["epochs"]
    except (OSError, ValueError, KeyError): total = None
    done = (int(ep[-1][0]) - (int(ep[-1][1]) - total)) if (ep and total) else None
    j = JOBS.get(n); job = None
    if j: job = {"kind": j["kind"], "running": j["proc"].poll() is None, "rc": j["proc"].poll(), "log": tail(j["log"]), "progress": progress.read(n, j["kind"])}
    sp = speakers.load(n)
    if sp and any(("pitch" not in p) or ("sample_ranges" not in p and p.get("sample_times")) for p in sp["people"]):
        try: sp = speakers.person_pitches(n) or sp          # older scan: add pitch + clip ranges once (a couple of seconds, no GPU)
        except Exception: pass
    return {"name": n, "base": m.get("base"), "base_used": m.get("base_used"),
            "people": sp["people"] if sp else None, "keep": m.get("keep_people") or [], "keep_from": m.get("keep_from"), "keep_until": m.get("keep_until"),
            "clean_voice": bool(m.get("clean_voice")), "people_names": m.get("people_names") or {}, "voice_type": m.get("voice_type") or "", "ref_yes": m.get("ref_yes") or [], "ref_no": m.get("ref_no") or [], "strict": m.get("strict") or "normal", "use_match": bool(m.get("use_match")),
            "match": ({k: sp["match"][k] for k in ("minutes", "stretches", "strict", "samples")} if sp and sp.get("match") else None), "people_stale": speakers.stale(n) if sp else False, "folder": str(src) if src else None, "folder_ok": bool(src and src.is_dir()),
            "files": [{"name": speakers.rel(n, p), "mb": round(p.stat().st_size / 1e6, 1)} for p in files],
            "clips": clips, "prepared": md.exists(), "edited": len(trim.pending(n)),
            "train": {"running": bool(train_pid(n)), "total": total, "done": max(0, done) if done is not None else None,
                      "mos": float(mos[-1]) if mos else None, "checkpoints": len(ck), "disk_gb": round(sum(p.stat().st_size for p in ck) / 2**30, 1), "log": tl.split("\n")[-3:] if tl else []},
            "job": job, "exported": (CFG.out / f"{n}.onnx").exists()}


def browse(path):
    """List the sub-folders of a folder for the page's folder chooser (folders only, never file contents)."""
    home = Path.home()
    p = Path(path).expanduser() if path else home
    try:
        p = p.resolve()
        if not p.is_dir(): p = p.parent if p.exists() else home
        entries = sorted(p.iterdir(), key=lambda c: c.name.lower())
        dirs = [c.name for c in entries if not c.name.startswith(".") and c.is_dir()]
        audio = sum(1 for c in entries if c.is_file() and c.suffix.lower() in AUDIO_EXT)
    except PermissionError:
        raise ValueError("You do not have permission to open that folder.")
    except OSError as e:
        raise ValueError(str(e))
    places = [str(x) for x in (home, home / "Music", home / "Downloads", home / "Documents", home / "Desktop", Path("/media"), Path("/mnt"),
                               Path("/run/media") / home.name) if x.is_dir()]
    return {"path": str(p), "parent": str(p.parent) if p != p.parent else None, "dirs": dirs[:1500], "audio": audio, "places": places}


class H(BaseHTTPRequestHandler):
    server_version = "VoiceStudio"
    def log_message(self, *a): pass

    def _guard(self):
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        if host not in CFG["allowed_hosts"]: self._err("bad host", 403); return False
        if self.command != "GET":
            if self.headers.get("X-Studio") != "1": self._err("forbidden", 403); return False
            o = self.headers.get("Origin")
            if o and urlparse(o).netloc != self.headers.get("Host"): self._err("forbidden", 403); return False
        return True

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def _err(self, msg, code=400): self._json({"error": str(msg)}, code)

    def _file(self, path, ctype, download=None):
        try: size = path.stat().st_size; f = open(path, "rb")
        except OSError: return self._err("not found", 404)
        start, end, code = 0, size - 1, 200
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-"); start = int(a or 0); end = int(b) if b else size - 1; code = 206
        self.send_response(code)
        if code == 206: self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Type", ctype); self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1)); self.send_header("Cache-Control", "no-store")
        if download: self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers(); f.seek(start); left = end - start + 1
        while left > 0:
            c = f.read(min(1 << 20, left))
            if not c: break
            self.wfile.write(c); left -= len(c)
        f.close()

    def _body(self):
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")

    def _dataset(self, n):
        """The prepared clips + transcript as one zip (what training reads): wavs/*.wav (22.05 kHz mono) + metadata.csv + a short README."""
        root = CFG.data / n; md = root / "metadata.csv"
        if not md.exists(): return self._err("Nothing is prepared yet.", 404)
        if trim.pending(n): return self._err("Some clips have edited words but were not reprocessed. Press 'Reprocess edited clips' first.", 409)
        names = [l.split("|", 1)[0] for l in md.read_text(encoding="utf-8").splitlines() if "|" in l]
        spool = tempfile.SpooledTemporaryFile(max_size=64 << 20)
        with zipfile.ZipFile(spool, "w", zipfile.ZIP_STORED) as z:
            z.write(md, "metadata.csv")
            for f in names:
                if (root / "wavs" / f).exists(): z.write(root / "wavs" / f, f"wavs/{f}")
            z.writestr("README.txt", "Dataset made with The Piper Mill.\n\nwavs/*.wav  mono, 16-bit, 22.05 kHz clips (1.5-12 s) of one speaker, levelled to the same loudness\n"
                       "metadata.csv  one line per clip:  file.wav|transcript   (the transcript is in spoken form, as Piper reads it)\n\n"
                       "Only use recordings of people who have agreed to it.\n")
        size = spool.tell(); spool.seek(0)
        self.send_response(200); self.send_header("Content-Type", "application/zip"); self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition", f'attachment; filename="{n}-dataset.zip"'); self.send_header("Cache-Control", "no-store"); self.end_headers()
        shutil.copyfileobj(spool, self.wfile); spool.close()

    def do_GET(self):
        if not self._guard(): return
        u = urlparse(self.path); q = parse_qs(u.query)
        try:
            if u.path == "/": return self._file(WEB / "index.html", "text/html; charset=utf-8")
            if u.path == "/api/browse": return self._json(browse(q.get("path", [""])[0]))
            if u.path == "/api/voices":
                b = busy()
                return self._json({"voices": [info(n) for n in voices.names()], "bases": {k: v[0] for k, v in base.BASES.items()},
                                   "busy": {"voice": b[0], "kind": b[1]} if b else None, "install": install.configured()})
            if u.path == "/api/transcript":
                n = CFG.check_name(q["voice"][0]); md = CFG.data / n / "metadata.csv"; rows = []
                if md.exists():
                    for line in open(md, encoding="utf-8"):
                        f, _, t = line.rstrip("\n").partition("|")
                        if f: rows.append({"file": f, "text": t})
                return self._json({"rows": rows})
            if u.path.startswith("/dataset/") and u.path.endswith(".zip"):
                return self._dataset(CFG.check_name(os.path.basename(unquote(u.path))[:-4]))
            if u.path.startswith("/audio/"):
                _, _, n, f = u.path.split("/", 3)
                return self._file(CFG.data / CFG.check_name(n) / "wavs" / os.path.basename(unquote(f)), "audio/wav")
            if u.path.startswith("/speaker-sample/"):
                _, _, n, f = u.path.split("/", 3)
                return self._file(CFG.runs / CFG.check_name(n) / "speakers" / os.path.basename(unquote(f)), "audio/wav")
            if u.path.startswith("/out/"):
                return self._file(CFG.out / os.path.basename(unquote(u.path[5:])), "audio/wav")
            if u.path.startswith("/download/"):
                f = os.path.basename(unquote(u.path[10:]))
                if not (f.endswith(".onnx") or f.endswith(".onnx.json")): return self._err("not found", 404)
                return self._file(CFG.out / f, "application/octet-stream", download=f)
        except (KeyError, ValueError) as e:
            return self._err(e)
        self._err("not found", 404)

    def do_POST(self):
        if not self._guard(): return
        u = urlparse(self.path)
        try:
            d = self._body()
            if u.path == "/api/voices":
                n = CFG.check_name(d.get("name")); key = d.get("base") or base.AUTO
                if key != base.AUTO and key not in base.BASES: raise ValueError("unknown starting voice")
                if voices.exists(n): raise ValueError(f"'{n}' already exists")
                if key != base.AUTO: base.ensure(key, log=lambda *_: None)
                folder = (d.get("folder") or "").strip()
                if not folder: raise ValueError("Choose the folder that has your recordings (use Browse).")
                if not Path(folder).expanduser().is_dir(): raise ValueError("That folder does not exist.")
                vt = d.get("voice_type") if d.get("voice_type") in ("male", "female") else ""
                voices.write(n, base=key, source_dir=str(Path(folder).expanduser().resolve()), voice_type=vt)
                return self._json({"ok": True, "license": (base.info(key) or {}).get("license", "") if key != base.AUTO else ""})
            n = CFG.check_name(d.get("voice"))
            if not voices.exists(n): raise ValueError("no such voice")
            if u.path == "/api/prep": start_job(n, "prep", ["prep", n])
            elif u.path == "/api/train":
                ep = max(1, min(20000, int(d.get("epochs", 1000))))
                if not (CFG.data / n / "metadata.csv").exists(): raise ValueError("Prepare the recordings first.")
                if trim.pending(n): raise ValueError(f"{len(trim.pending(n))} clip(s) have edited words but their audio was not cut to match. Press 'Reprocess edited clips' in Check the words first.")
                start_job(n, "train", ["train", n, str(ep)], meta={"epochs": ep})
            elif u.path == "/api/match":
                def ranges(v):
                    out = []
                    for x in v or []:
                        if len(x) == 3 and isinstance(x[0], str): out.append([x[0], float(x[1]), float(x[2])])
                    return out[:300]
                yes, no = ranges(d.get("yes")), ranges(d.get("no"))
                if not yes: raise ValueError("Tick at least one clip (this is my speaker) first.")
                if not speakers.load(n): raise ValueError("Find the speakers first.")
                strict = d.get("strict") if d.get("strict") in speakers.STRICT else "normal"
                voices.write(n, ref_yes=yes, ref_no=no, strict=strict)
                start_job(n, "match", ["match", n])
            elif u.path == "/api/reset":
                if busy() or train_pid(n): raise RuntimeError("A job is running. Wait for it or stop it first.")
                start_over(n, training=bool(d.get("training")))
            elif u.path == "/api/personname":
                r = speakers.load(n)
                try: pid = int(d.get("id"))
                except (TypeError, ValueError): raise ValueError("Unknown person.")
                if not r or pid not in {p["id"] for p in r["people"]}: raise ValueError("Unknown person.")
                names = dict(voices.read(n).get("people_names") or {}); name = " ".join(str(d.get("name") or "").split())[:40]
                if name: names[str(pid)] = name
                else: names.pop(str(pid), None)                    # blank = back to "Person N"
                voices.write(n, people_names=names)
            elif u.path == "/api/voicetype":
                vt = d.get("type") if d.get("type") in ("male", "female") else ""
                voices.write(n, voice_type=vt)
                if vt and speakers.load(n):
                    try: speakers.person_pitches(n)          # a few seconds, no GPU: so the list can show who fits
                    except Exception: pass
            elif u.path == "/api/usematch":
                r = speakers.load(n)
                if d.get("on") and not (r and r.get("match")): raise ValueError("Nothing has been matched yet.")
                voices.write(n, use_match=bool(d.get("on")))
            elif u.path == "/api/trim":
                if not trim.pending(n): raise ValueError("No clip's words have been changed, so there is nothing to reprocess. Save your edits first.")
                start_job(n, "trim", ["trim", n])
            elif u.path == "/api/speakers":
                num = int(d["num"]) if str(d.get("num") or "").strip().isdigit() else None
                start_job(n, "speakers", ["speakers", n] + (["--num", str(num)] if num else []))
            elif u.path == "/api/keep":
                ids = [int(i) for i in d.get("ids", [])]
                if ids: speakers.keep(n, ids)
                else: voices.write(n, keep_people=[])
                voices.write(n, use_match=False)                       # saving ticked people means: use the people, not a match
            elif u.path == "/api/window":
                def secs(v):
                    v = str(v or "").strip()
                    if not v: return None
                    parts = [float(x) for x in v.split(":")]
                    if len(parts) > 3 or any(x < 0 for x in parts): raise ValueError("Times look like 21:30 (minutes:seconds) or 1:05:00.")
                    return sum(x * 60 ** i for i, x in enumerate(reversed(parts)))
                try: a, b = secs(d.get("from")), secs(d.get("until"))
                except ValueError: raise ValueError("Times look like 21:30 (minutes:seconds) or 1:05:00.")
                if a is not None and b is not None and b <= a: raise ValueError("The end time must be after the start time.")
                voices.write(n, keep_from=a, keep_until=b, clean_voice=bool(d.get("clean")))
            elif u.path == "/api/job/stop":
                running = job_pids(n)
                if not running: raise ValueError("Nothing is running for this voice.")
                for _, pid in running:
                    try: os.killpg(os.getpgid(pid), signal.SIGTERM)
                    except OSError: pass
            elif u.path == "/api/train/stop":
                pid = train_pid(n)
                if pid: os.killpg(os.getpgid(pid), signal.SIGINT)
            elif u.path == "/api/export":
                which = d.get("which") if d.get("which") in ("best", "accurate", "latest") else "best"
                start_job(n, "export", ["export", n, "--text", (d.get("text") or "")[:400], "--which", which])
            elif u.path == "/api/clean":
                if busy(): raise RuntimeError("A job is running. Wait for it or stop it first.")
                return self._json({"ok": True, "freed_gb": round(train_mod.prune(n), 1)})
            elif u.path == "/api/source":
                p = (d.get("path") or "").strip()
                if not p or not Path(p).expanduser().is_dir(): raise ValueError("That folder does not exist.")
                voices.write(n, source_dir=str(Path(p).expanduser().resolve()))
                speakers.result_path(n).unlink(missing_ok=True); voices.write(n, keep_people=[], people_names={})   # a new folder means new speakers
            elif u.path == "/api/install":
                if not install.configured(): raise ValueError("Install is not set up. Use the download buttons and copy the two files to your Piper voices folder.")
                r = subprocess.run([sys.executable, "-m", "voicestudio", "install", n], capture_output=True, text=True, timeout=300)
                if r.returncode: raise RuntimeError((r.stdout + r.stderr)[-600:])
                return self._json({"ok": True, "message": r.stdout.strip()[-300:]})
            else: return self._err("not found", 404)
            return self._json({"ok": True})
        except (ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired, OSError, SystemExit) as e:
            self._err(e)

    def do_PUT(self):
        if not self._guard(): return
        u = urlparse(self.path); q = parse_qs(u.query)
        try:
            if u.path == "/api/transcript":
                d = self._body(); n = CFG.check_name(d["voice"]); md = CFG.data / n / "metadata.csv"
                if not md.exists(): raise ValueError("nothing prepared yet")
                shutil.copy(md, md.with_suffix(".csv.bak"))
                if not (md.parent / "metadata.orig.csv").exists(): shutil.copy(md, md.parent / "metadata.orig.csv")   # first edit: remember the transcript as it was
                wavs = CFG.data / n / "wavs"; lines = []
                for r in d["rows"]:
                    f = os.path.basename(r["file"]); t = " ".join(str(r["text"]).split()).replace("|", " ")
                    if t and (wavs / f).exists(): lines.append(f"{f}|{t}")
                md.write_text("\n".join(lines) + "\n", encoding="utf-8")
                shutil.rmtree(CFG.cache / n, ignore_errors=True)              # Piper's cache is keyed by row+text: start fresh
                return self._json({"ok": True, "kept": len(lines), "edited": len(trim.pending(n))}) 
            self._err("not found", 404)
        except (ValueError, KeyError, OSError) as e:
            self._err(e)

    def do_DELETE(self):
        if not self._guard(): return
        u = urlparse(self.path); q = parse_qs(u.query)
        try:
            n = CFG.check_name(q["voice"][0])
            if u.path == "/api/voice":
                if busy(): raise RuntimeError("A job is running. Wait for it or stop it first.")
                voices.delete(n); return self._json({"ok": True})
            self._err("not found", 404)
        except (ValueError, KeyError, RuntimeError, OSError) as e:
            self._err(e)


def serve():
    CFG.ensure_dirs()
    host, port = CFG["host"], int(CFG["port"])
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("WARNING: listening beyond this computer. There is no login. Only do this on a network you trust.", flush=True)
    print(f"The Piper Mill on http://{host}:{port}   (data: {CFG.home})", flush=True)
    ThreadingHTTPServer((host, port), H).serve_forever()

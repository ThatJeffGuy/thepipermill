"""Reading any audio file safely: pick the right track, handle stereo / surround / odd sample rates, report problems clearly."""
import json, re, subprocess
from pathlib import Path
from .config import CFG


class AudioError(Exception):
    """A file that cannot be used, with a message a person can read."""


def _why(stderr, fallback):
    """The useful last line of ffmpeg's complaint, without file paths and internal tags."""
    line = (stderr.strip().splitlines() or [fallback])[-1]
    line = re.sub(r"^\[[^\]]*\]\s*", "", line)                    # [out#0/wav @ 0x...]
    line = re.sub(r"^(/|[A-Za-z]:\\)[^:]*:\s*", "", line)           # /path/to/file.mp3:
    line = line.replace("Invalid data found when processing input", "not a valid audio file")
    line = line.replace("Nothing was written into output file, because at least one of its streams received no packets.", "no audio could be read from it")
    return line[:100]


def probe(src: Path):
    """Audio streams of a file: [{'index','channels','layout','lang'}]. Raises AudioError if unreadable or silent-less (no audio)."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                        "stream=index,channels,channel_layout:stream_tags=language", "-of", "json", str(src)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise AudioError(f"cannot be read ({_why(r.stderr, 'unreadable')})")
    streams = json.loads(r.stdout or "{}").get("streams", [])
    if not streams: raise AudioError("has no audio track")
    return [{"index": s["index"], "channels": s.get("channels", 1), "layout": s.get("channel_layout", ""),
             "lang": (s.get("tags") or {}).get("language", "")} for s in streams]


def choose_track(streams):
    """Position (among audio streams) of the track to use: your preferred language if present, else the first."""
    want = (CFG["audio_language"] or "").strip().lower()
    if want:
        for k, s in enumerate(streams):
            if s["lang"].lower().startswith(want[:3]): return k
    return 0


def _mono_filter(stream):
    # surround recordings keep dialogue in the centre channel: use it alone instead of a muddy mix of all channels
    if stream["channels"] >= 5 and any(stream["layout"].startswith(x) for x in ("5.", "7.")): return "pan=mono|c0=FC"
    return None


def decode(src: Path, dst: Path, sr: int, loudnorm=False, log=print):
    """Any audio/video file -> mono 16-bit wav at `sr`. Returns seconds of audio. Raises AudioError."""
    streams = probe(src)
    k = choose_track(streams); st = streams[k]
    if len(streams) > 1:
        log(f"    {src.name}: {len(streams)} audio tracks, using #{k + 1} ({st['lang'] or 'unknown language'}, {st['channels']} ch)")
    chain = [f for f in (_mono_filter(st), "highpass=f=60", "loudnorm=I=-20:TP=-2:LRA=11" if loudnorm else None) if f]
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-vn", "-map", f"0:a:{k}", "-ac", "1", "-ar", str(sr)]
    if chain: cmd += ["-af", ",".join(chain)]
    cmd += ["-sample_fmt", "s16", str(dst)]
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not dst.exists() or dst.stat().st_size < 1000:
        dst.unlink(missing_ok=True)
        raise AudioError(f"could not be decoded ({_why(r.stderr, 'decoding failed')})")
    seconds = (dst.stat().st_size - 44) / (2 * sr)
    if seconds < 1.0:
        dst.unlink(missing_ok=True); raise AudioError("is shorter than one second")
    return seconds

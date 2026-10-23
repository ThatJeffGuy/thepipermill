"""Runs INSIDE the separate "clone" environment (.venv-clone), never in the main one.

Reads a job file, then: (1) speaks every sentence in the reference voice with the cloner (OmniVoice by default, F5-TTS as the alternative), (2) writes down what was actually said with Whisper,
(3) measures how much each clip sounds like the reference (ECAPA speaker voiceprint), (4) keeps the clips whose words are right and whose
voice matches, up to the requested minutes. Output: <out_dir>/kept/NNNN.wav + kept.csv (`file.wav|text`) + report.json.
It is a plain script on purpose (standard library + the clone environment only) so the main program never has to import the cloner.
"""
import json, os, re, sys, time, types, wave
from pathlib import Path


def _progress(out, stage, done, total, note=""):
    p = Path(out) / "progress.json"; tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"stage": stage, "done": done, "total": total, "note": note[:90], "t": time.time()})); os.replace(tmp, p)


def _words(t): return re.sub(r"[^a-z0-9' ]", " ", t.lower().replace("-", " ")).split()


def wer(ref, hyp):
    r, h = _words(ref), _words(hyp)
    if not r: return 1.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1])); prev, d[j] = d[j], cur
    return d[len(h)] / len(r)


def main(job_path):
    job = json.load(open(job_path)); out = Path(job["out_dir"]); raw = out / "raw"; kept_dir = out / "kept"
    raw.mkdir(parents=True, exist_ok=True); kept_dir.mkdir(parents=True, exist_ok=True)
    dev = job.get("device") or "cuda"
    # The F5 package imports `datasets` (only needed for its own training code) and `datasets` needs pyarrow, which crashes on some
    # systems. We never train F5, so a stand-in is enough.
    stub = types.ModuleType("datasets"); stub.Dataset = object; stub.load_from_disk = None; sys.modules["datasets"] = stub
    import numpy as np, soundfile as sf, torch
    if dev == "cuda" and torch.cuda.is_available() and job.get("gpu_fraction"): torch.cuda.set_per_process_memory_fraction(float(job["gpu_fraction"]))
    S, target_s = job["sentences"], job["target_minutes"] * 60
    # ---- 1. speak the sentences (resumable: clips already on disk are kept)
    engine = job.get("engine", "omnivoice")
    if engine == "f5":
        from f5_tts.api import F5TTS
        cloner = F5TTS(device=dev)
        def speak(t, i):
            wav, sr, _ = cloner.infer(ref_file=job["ref_wav"], ref_text=job["ref_text"], gen_text=t, seed=i, nfe_step=job.get("nfe_step", 32),
                                      show_info=lambda *a: None, progress=None)
            return wav, sr
    else:
        from omnivoice import OmniVoice
        cloner = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0" if dev == "cuda" else "cpu", dtype=torch.float16 if dev == "cuda" else torch.float32)
        def speak(t, i):
            torch.manual_seed(i)
            return cloner.generate(text=t, ref_audio=job["ref_wav"], ref_text=job["ref_text"], speed=float(job.get("speed", 1.0)))[0], 24000
    have = {p.name for p in raw.glob("*.wav")}; total_s = 0.0; meta = {}
    mp = out / "raw.json"
    if mp.exists(): meta = json.loads(mp.read_text())
    for i, t in enumerate(S):
        fn = f"{i:05d}.wav"
        if fn in have:
            total_s += meta.get(fn, {}).get("sec", 0)
        else:
            if total_s >= target_s * job.get("overshoot", 1.5): break
            try: wav, sr = speak(t, i)
            except Exception as e:
                print("skip", i, repr(e)[:120], flush=True); continue
            sf.write(raw / fn, wav, sr); meta[fn] = {"text": t, "sec": len(wav) / sr}; total_s += len(wav) / sr
            if i % 10 == 0: mp.write_text(json.dumps(meta))
        _progress(out, "Speaking the sentences in the reference voice", int(total_s), int(target_s * job.get("overshoot", 1.5)), f"{i + 1} sentences")
    mp.write_text(json.dumps(meta)); files = sorted(meta)
    del cloner
    if dev == "cuda": torch.cuda.empty_cache()
    # ---- 2. what was really said
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model=job["whisper_model"], device=0 if dev == "cuda" else -1,
                   torch_dtype=torch.float16 if dev == "cuda" else torch.float32)
    heard = {}; gen = {"task": "transcribe", "language": job.get("language", "en"), "num_beams": 1}
    for k, fn in enumerate(files):
        a, sr = sf.read(raw / fn, dtype="float32"); a = a if a.ndim == 1 else a.mean(1)
        if sr != 16000:
            import librosa; a = librosa.resample(a, orig_sr=sr, target_sr=16000)
        try: heard[fn] = asr({"raw": a, "sampling_rate": 16000}, generate_kwargs=gen)["text"].strip()
        except ValueError:                                        # an English-only Whisper model takes no language/task options
            gen = {"num_beams": 1}; heard[fn] = asr({"raw": a, "sampling_rate": 16000}, generate_kwargs=gen)["text"].strip()
        if k % 10 == 0: _progress(out, "Checking the words", k, len(files))
    del asr
    if dev == "cuda": torch.cuda.empty_cache()
    # ---- 3. does it still sound like the reference?
    from speechbrain.inference.speaker import EncoderClassifier
    enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(Path(job["cache_dir"]) / "ecapa"), run_opts={"device": dev})
    import torchaudio
    def emb(p):
        a, sr = sf.read(p, dtype="float32"); a = a if a.ndim == 1 else a.mean(1)
        t = torchaudio.functional.resample(torch.tensor(a), sr, 16000)[None].to(dev)
        return torch.nn.functional.normalize(enc.encode_batch(t).squeeze().cpu(), dim=0)
    ref = emb(job["ref_wav"]); sim = {}
    for k, fn in enumerate(files):
        sim[fn] = float(ref @ emb(raw / fn))
        if k % 10 == 0: _progress(out, "Checking that it sounds like the reference", k, len(files))
    # ---- 4. keep the good ones
    w = {fn: wer(meta[fn]["text"], heard[fn]) for fn in files}
    floor = max(job.get("min_sim", 0.35), float(np.percentile(list(sim.values()), job.get("drop_worst_pct", 15)))) if sim else 0
    good = [fn for fn in files if w[fn] <= job.get("max_wer", 0.15) and sim[fn] >= floor and 1.2 <= meta[fn]["sec"] <= 14]
    rows, tot, n = [], 0.0, 0
    for fn in good:
        if tot + meta[fn]["sec"] > target_s: break
        n += 1; name = f"{n:05d}.wav"; (kept_dir / name).write_bytes((raw / fn).read_bytes()); rows.append(f"{name}|{meta[fn]['text']}"); tot += meta[fn]["sec"]
    (out / "kept.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    rep = {"generated": len(files), "kept": len(rows), "minutes": round(tot / 60, 1), "similarity_floor": round(floor, 3),
           "mean_similarity": round(float(np.mean(list(sim.values()))), 3) if sim else None, "mean_wer": round(float(np.mean(list(w.values()))), 3) if w else None,
           "dropped_words": sum(1 for fn in files if w[fn] > job.get("max_wer", 0.15)), "dropped_voice": sum(1 for fn in files if sim[fn] < floor)}
    (out / "report.json").write_text(json.dumps(rep, indent=1)); _progress(out, "done", 1, 1, f"{len(rows)} clips, {rep['minutes']} min"); print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

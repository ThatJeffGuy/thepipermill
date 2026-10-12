# The Piper Mill

Train your own [Piper](https://github.com/OHF-Voice/piper1-gpl) text-to-speech voice from your own recordings, using a simple web page on your own computer.

*An independent tool for making Piper voices. It is not affiliated with or endorsed by the Piper project or the Open Home Foundation. Formerly called Voice Studio: the command, folders and service are still named `voicestudio` / `voice-studio`.*

1. Point it at a folder of your recordings.
2. It cuts them into clips and writes down what you said.
3. You read through the words and fix any mistakes.
4. It trains on your GPU, with a progress bar.
5. You listen to a test clip and download `yourvoice.onnx` + `yourvoice.onnx.json`, ready for Piper, Home Assistant and other tools that use Piper voices.

Everything runs locally. Nothing is uploaded anywhere (the only downloads are the tools, the starting voice and the Whisper model).

> **Use only your own voice, or the voice of someone who has agreed to it.** Do not train on recordings of other people (actors, public figures, friends) without their permission. Voice cloning of real people without consent can cause real harm and may be illegal where you live.

## What you need

- Linux with an **NVIDIA GPU** (8 GB+ VRAM recommended; training will not start on a CPU) and a working driver (`nvidia-smi`)
- `git`, `ffmpeg`, a C/C++ compiler (`sudo apt install git ffmpeg build-essential`)
- Python 3.11 or 3.12, or [`uv`](https://docs.astral.sh/uv/) (it fetches Python for you; recommended)
- About 15 GB of free disk space (PyTorch is large), plus space for your recordings

Tested on Ubuntu with an RTX 5070 (Blackwell, CUDA 13 build of PyTorch).

## Install

```bash
git clone https://github.com/ThatJeffGuy/thepipermill.git && cd thepipermill
./install.sh          # creates .venv, installs PyTorch and the Piper trainer, checks the GPU
./run.sh              # then open http://127.0.0.1:8765
```

Optional: `scripts/install-service.sh` starts it automatically when you log in.

## Using it

Open the page and press **+ New voice**. Give it a short name, choose the folder with your recordings, and say whether the voice is **male, female or not specified**. That choice helps find your voice in a mixed recording and picks the starting voice (it is only the starting point: the result learns your voice from your recordings).

**One step at a time.** The page is a wizard: **Prev** / **Next**, and a **Skip** button on the optional steps (Speakers, Refine, Words). Next unlocks when a step is done, and the message beside it says what is missing. While something runs, that message pulses (for example "Finding the speakers…"), the bar shows real progress with a time estimate, and the footer says what the GPU is busy with. A strip at the top shows where you are: green ✓ done, solid blue current, pulsing red working, dashed dash skipped, grey not reached yet.

The **Advanced** button at the top only reveals extras: the console with each job's log, Voice settings (starting voice, free up disk space, delete the voice), the version choice when you listen, the "how many people talk" box, the starting-voice picker when you create a voice, and free jumping between steps. If a job fails, everyone gets a plain one-line reason.

Every long job has a **Stop** button. **Start over** (bottom of the page) clears a voice's speaker scan, choices and prepared clips so you can begin again; your recordings are never touched and training is only deleted if you choose that too.

| Step | What to do |
|---|---|
| 1. Recordings folder | Choose (Browse…) the folder on your computer that holds your recordings; they are read from there and never copied or changed. 30–60+ minutes works best. |
| 2. Who is speaking? *(optional)* | Only if your recordings have more than one person (a conversation, a call, a film). Finds each voice, shows samples with the time each comes from, and lets you tick whose voice to train on. Tick ✓ the sample clips that are your speaker. See below. |
| 3. Refine *(optional)* | **Find more like my voice**: looks through the whole recording for more speech that sounds like the clips you ticked, shows how many minutes it found, and lets you tick or cross more and repeat. When a round finds nothing new, it tells you your voice has settled and to go on. |
| 4. Prepare | One click. Splits at pauses and transcribes with Whisper on the GPU. Uses only the people you chose (or the voice you matched). Tick **Remove music and background noise** to keep only the voices (see below). |
| 5. Check the words *(optional)* | Press ▶ on a clip, fix the text, remove bad clips (✕), save. **The voice learns exactly what is written here, so skipping this may reduce quality.** Words from another person can be cut out with **Reprocess**, below. **Export dataset (.zip)** saves the cleaned clips and transcript (22.05 kHz mono wavs + `metadata.csv`) to use anywhere. |
| 6. Train | Quick / Normal / Long. It keeps running if you close the tab; stop and resume any time. The **quality score** (1 to 5, hover it for details) is a predicted listener rating of how natural the test sentences sound; it does not measure how much the voice sounds like you. The progress bar can take 10–20 minutes to update for the first time (it moves when training saves a new checkpoint), so do not stop training until it has updated at least once. |
| 7. Listen | Type a sentence, make a test clip, then download the two voice files. It uses the best-sounding version; **Advanced** lets you choose another. |

**Recording tips:** quiet room, same distance from the microphone, no music or fans, no clipping. Read or talk naturally and vary it: questions, short and long sentences, numbers. The model copies how you speak, not just your tone.

### Tuned for Piper

The Piper Mill prepares and trains the way Piper wants it:
- **Audio:** 22.05 kHz mono, clips of 1.5 to 12 seconds cut at natural pauses, every clip levelled to the same loudness (Piper trims leading and trailing silence itself).
- **Text:** transcripts are converted to how the words are *spoken*, because Piper reads text with espeak-ng: "15%" becomes "fifteen percent", "8.30" after "at" becomes "eight thirty", "$3.50" becomes "three dollars and fifty cents", "Dr." becomes "doctor". Repeated-phrase hallucinations and junk are dropped. Check the result in step 4.
- **Training:** fine-tunes from a Piper base voice chosen to match your recordings, so 30 to 60 minutes is enough. Piper holds out 10% of your clips to score progress. If you edit or re-prepare, its cache is refreshed automatically so old audio is never reused.
- **Which version you get:** Piper keeps its best checkpoints by sound quality and by accuracy, plus the latest. The test clip and the downloaded voice default to the **best-sounding** one by quality score, which is usually better than simply the newest. You can pick *Most accurate* or *Latest* in step 6.
- **Disk space:** every checkpoint is about 0.8 GB, so one voice can use up to about 9 GB. *Voice settings → Free up space* keeps only the best and the latest (or `voicestudio clean myvoice`).
- **Output:** `name.onnx` + `name.onnx.json`, the standard Piper voice format.

### Real-world recordings

Prepare and "Who is speaking?" are built for messy folders, not just tidy ones:
- **Long files and big folders:** hours of audio and hundreds of files are fine, with a progress bar and time estimate. No clip is ever longer than 15 seconds (long stretches are cut at the quietest moment).
- **Noisy rooms:** the silence threshold is chosen per recording, and you get a note when a recording is noisy.
- **Any format:** wav, mp3, m4a, flac, ogg, opus, and video files (mp4, mkv, webm). Mono, stereo and surround (surround uses the centre channel, where speech lives) at any sample rate. For videos with several audio tracks, set `audio_language` (for example `"eng"`) in the config; otherwise the first track is used.
- **Files that cannot be used** (corrupt, empty, too short, no speech) are skipped and listed at the end. They never stop the run.

### Recordings with more than one person (optional step 2)

Step 2 uses [pyannote.audio](https://github.com/pyannote/pyannote-audio) speaker diarization (the open `pyannote/speaker-diarization-community-1` model) to find who speaks when. Each voice it finds is listed with how much they speak, when they are **first heard**, and their typical pitch. Sort the list by *Most speech* or *First heard*. Two speakers the model separated inside one recording are never merged; across several recordings, the same person is matched by voice fingerprint (lower `speaker_match_distance` in the config if two people get merged).

Each person has **▶ samples spread through the recording** (the number is the time each comes from), so you hear typical clips and not only the clearest. Tick the person to train on and press **Save my choice**.

**Looking for a male or female voice:** set it in the New voice box or next to the speaker count. Voices clearly in the wrong pitch range are dimmed ("probably not a male voice"), and the matching below skips them. Pitch only rules out obviously wrong voices; it cannot tell two similar men apart.

**Find more like my voice:** voices can still be mixed up (a person may contain your speaker and similar-sounding others). Press **✓** on sample clips that are your speaker (and **✗** on ones that are not), choose Strict / Normal / Loose, and press *Find more like these*. It compares every stretch of speech in the whole recording with your ticked clips, across all the people found, and shows how many minutes it found plus new samples to check (the ones marked **?** are the least sure). Tick or cross more and run it again to improve it, then choose *Train on the matched voice* or *the ticked people instead*. Two or more ticked clips work best; stretches under one second are skipped because they are too short to recognise reliably.

That model is **gated on Hugging Face**, so once only:
1. Log in at <https://huggingface.co>, open <https://huggingface.co/pyannote/speaker-diarization-community-1> and accept its conditions.
2. Create a read token at <https://huggingface.co/settings/tokens>.
3. Put it in your config as `hf_token = "..."`, or set `HF_TOKEN`, or run `.venv/bin/hf auth login`.

Only use recordings of people who have agreed to it. Telling the tool how many people talk ("auto" by default) improves accuracy.

### Removing music and background noise (Prepare)

For films, videos and noisy recordings, tick **Remove music and background noise** in Prepare. It uses [Demucs](https://github.com/facebookresearch/demucs) to keep only the voices, and it runs only on the parts Prepare keeps (your chosen speaker or matched voice), so it adds very little time. The clips and their transcripts are then made from the cleaned audio. The first use downloads a small model (about 80 MB). A clean recording of just you does not need it, and it is off by default. From the command line: `voicestudio prep myvoice --clean`.

### Two voices in one clip: Reprocess (step 4)

Prepare cuts at pauses, so a clip can hold two people talking. In *Check the words*, delete the words the wrong person says from the **start or end** of the clip's text, press **Save**, then **Reprocess edited clips**. The Piper Mill listens to each edited clip again with word-by-word timing and cuts the audio down to just the words you kept. Spelling-only fixes are left alone. If words are removed from the **middle**, the clip cannot be cut and you are told to delete it instead. The untrimmed originals are kept in `wavs.untrimmed/`, and training will not start while edited clips are waiting to be reprocessed, so text and audio always match.

### Command line

Everything the page does is also available as `.venv/bin/voicestudio`:

```
voicestudio new myvoice --folder ~/recordings   # create (add --base ryan|amy|lessac|joe|john to choose the starting voice)
voicestudio speakers myvoice      # optional: who speaks in the recordings; then: voicestudio keep myvoice 1
voicestudio match myvoice         # optional: find more speech like the clips you ticked in the page
voicestudio prep myvoice          # clips + transcripts  -> check data/myvoice/metadata.csv
voicestudio trim myvoice          # reprocess: cut edited clips down to the words you kept
voicestudio train myvoice 1000    # extra epochs, resumable
voicestudio export myvoice        # -> out/myvoice.onnx, out/myvoice.onnx.json, out/myvoice-test.wav  (--which best|accurate|latest)
voicestudio clean myvoice         # free disk space: keep only the best and latest checkpoints
voicestudio list | doctor | delete myvoice
```

### Where things are

Default data folder: `~/voice-studio-data` with `data/` (prepared clips), `runs/` (training checkpoints), `out/` (finished voices) and `base/` (starting voices). Change it in the config.

### Using your voice with Piper

Put both files (same name, `.onnx` and `.onnx.json`) in the folder your Piper reads voices from, then restart or reload Piper.
To automate that, set `[install]` in the config (copy to a local folder, `scp` to another machine, or run your own script). See [config.example.toml](config.example.toml).

## Settings

Copy `config.example.toml` to `~/.config/voice-studio/config.toml`. Every option is optional (data folder, port, language, Whisper model, batch size, install step, protected voice names).

The page listens on `127.0.0.1` only and has **no login**. Do not expose it to a network you do not trust. It also rejects requests that do not come from the page itself, so other websites cannot drive it.

## Licences and credits

- The Piper Mill itself: MIT (see `LICENSE`).
- It installs and drives [piper1-gpl](https://github.com/OHF-Voice/piper1-gpl) (GPL-3.0) as a separate program; it is downloaded by `install.sh` and not included here.
- Starting voices come from [rhasspy/piper-checkpoints](https://huggingface.co/datasets/rhasspy/piper-checkpoints). **Each has its own dataset licence, often non-commercial** (for example Ryan is CC BY-NC-SA 4.0). A voice you train from one inherits those limits. The licence is shown when you create a voice and saved next to the file in `base/`.
- Transcription uses OpenAI Whisper (MIT) through Hugging Face `transformers`.
- Removing background noise uses Demucs (MIT) by Meta Research.
- Speaker finding uses pyannote.audio (MIT) with the `speaker-diarization-community-1` model (CC-BY-4.0, gated: accept its conditions on Hugging Face).

## Troubleshooting

- **"CUDA is not available"**: run `nvidia-smi`; reinstall the driver, then `./install.sh` again. Very new GPUs need a PyTorch built for CUDA 12.8 or newer (the default install is).
- **Out of memory while training**: set `batch_size = 8` (or 4) in the config.
- **Whisper is slow or runs out of memory**: set `whisper_model = "openai/whisper-small"`.
- **Transcripts have mistakes**: normal. Fix them in *Check the words* (step 4); it matters more than anything else for quality.
- **Clips are of the wrong person**: pick the right person in step 2, use **Find more like my voice** (step 3), and use **Reprocess** for clips that mix two people. **Start over** clears it all if you want a clean run.
- **The speaker scan seems stuck**: the bar shows real progress; a long recording can take several minutes. **Stop** cancels it.
- **Voice sounds robotic or mumbles**: more (and cleaner) audio, and more training rounds. Listen to test clips as it trains and stop when it sounds right; more is not always better.

## Developing

`scripts/sanitize_check.sh` scans the repository for private addresses, usernames, tokens and stray models or audio. Run it before publishing.

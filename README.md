# Voice Studio

Train your own [Piper](https://github.com/OHF-Voice/piper1-gpl) text-to-speech voice from your own recordings, using a simple web page on your own computer.

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
git clone https://github.com/ThatJeffGuy/voice-studio.git && cd voice-studio
./install.sh          # creates .venv, installs PyTorch and the Piper trainer, checks the GPU
./run.sh              # then open http://127.0.0.1:8765
```

Optional: `scripts/install-service.sh` starts it automatically when you log in.

## Using it

Open the page and press **+ New voice**. Pick a starting voice that is close to yours (male or female): it is only the starting point, the result learns your voice from your recordings.

| Step | What to do |
|---|---|
| 1. Recordings folder | Choose (Browse…) the folder on your computer that holds your recordings; they are read from there and never copied or changed. 30–60+ minutes works best. |
| 2. Who is speaking? *(optional)* | Only if your recordings have more than one person (a conversation, a call). Finds each voice, groups the same person across files, and lets you listen to samples and tick whose voice to train on. See below. |
| 3. Prepare | One click. Splits at pauses and transcribes with Whisper on the GPU. If you picked speakers in step 2, only their parts are used. |
| 4. Check the words | Press ▶ on a clip, fix the text, remove bad clips (✕), save. **The voice learns exactly what is written here.** |
| 5. Train | Quick / Normal / Long. It keeps running if you close the tab; stop and resume any time. |
| 6. Listen | Type a sentence, make a test clip, then download the two voice files. |

**Recording tips:** quiet room, same distance from the microphone, no music or fans, no clipping. Read naturally and vary it: questions, short and long sentences, numbers. The model copies how you speak, not just your tone. There is a list of sentences to read in the page ("Need something to read?").

### Recordings with more than one person (optional step 2)

Step 2 uses [pyannote.audio](https://github.com/pyannote/pyannote-audio) speaker diarization (the open `pyannote/speaker-diarization-community-1` model) to find who speaks when, then matches the same person across your files by comparing voice fingerprints. You listen to a couple of samples of each person and tick the ones to train on; everyone else is left out of the dataset. If a person is split into two entries, tick both. If two people are merged, lower `speaker_match_distance` in the config and run it again.

That model is **gated on Hugging Face**, so once only:
1. Log in at <https://huggingface.co>, open <https://huggingface.co/pyannote/speaker-diarization-community-1> and accept its conditions.
2. Create a read token at <https://huggingface.co/settings/tokens>.
3. Put it in your config as `hf_token = "..."`, or set `HF_TOKEN`, or run `.venv/bin/hf auth login`.

Only use recordings of people who have agreed to it. Telling the tool how many people talk ("auto" by default) improves accuracy. From the command line: `voicestudio speakers myvoice [--num 2]`, then `voicestudio keep myvoice 1` (or `--all`).

### Command line

Everything the page does is also available as `.venv/bin/voicestudio`:

```
voicestudio new myvoice --folder ~/recordings   # create (add --base ryan|amy|lessac|joe|john to choose the starting voice)
voicestudio speakers myvoice      # optional: who speaks in the recordings; then: voicestudio keep myvoice 1
voicestudio prep myvoice          # clips + transcripts  -> check data/myvoice/metadata.csv
voicestudio train myvoice 1000    # extra epochs, resumable
voicestudio export myvoice        # -> out/myvoice.onnx, out/myvoice.onnx.json, out/myvoice-test.wav
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

- Voice Studio itself: MIT (see `LICENSE`).
- It installs and drives [piper1-gpl](https://github.com/OHF-Voice/piper1-gpl) (GPL-3.0) as a separate program; it is downloaded by `install.sh` and not included here.
- Starting voices come from [rhasspy/piper-checkpoints](https://huggingface.co/datasets/rhasspy/piper-checkpoints). **Each has its own dataset licence, often non-commercial** (for example Ryan is CC BY-NC-SA 4.0). A voice you train from one inherits those limits. The licence is shown when you create a voice and saved next to the file in `base/`.
- Transcription uses OpenAI Whisper (MIT) through Hugging Face `transformers`.
- Speaker finding uses pyannote.audio (MIT) with the `speaker-diarization-community-1` model (CC-BY-4.0, gated: accept its conditions on Hugging Face).

## Troubleshooting

- **"CUDA is not available"**: run `nvidia-smi`; reinstall the driver, then `./install.sh` again. Very new GPUs need a PyTorch built for CUDA 12.8 or newer (the default install is).
- **Out of memory while training**: set `batch_size = 8` (or 4) in the config.
- **Whisper is slow or runs out of memory**: set `whisper_model = "openai/whisper-small"`.
- **Transcripts have mistakes**: normal. Fix them in step 3; it matters more than anything else for quality.
- **Voice sounds robotic or mumbles**: more (and cleaner) audio, and more training rounds. Listen to test clips as it trains and stop when it sounds right; more is not always better.

## Developing

`scripts/sanitize_check.sh` scans the repository for private addresses, usernames, tokens and stray models or audio. Run it before publishing.

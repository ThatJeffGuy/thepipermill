# The Piper Mill

Train your own [Piper](https://github.com/OHF-Voice/piper1-gpl) text-to-speech voice from your own recordings, using a simple web page on your own computer.

*A project of [donutmoneyit.ca](https://donutmoneyit.ca). An independent tool for making Piper voices. It is not affiliated with or endorsed by the Piper project or the Open Home Foundation. Formerly called Voice Studio: the command, folders and service are still named `voicestudio` / `voice-studio`.*

1. Point it at a folder of your recordings.
2. It cleans the audio first (music and background out, hiss reduced, voice boosted, one even volume), then cuts it into clips and writes down what you said.
3. You read through the transcripts and fix any mistakes.
4. It trains on your GPU, with a progress bar.
5. You listen to a test clip and download `yourvoice.onnx` + `yourvoice.onnx.json`, ready for Piper, Home Assistant and other tools that use Piper voices.

The web page is built for a desktop screen: it needs a browser window of at least 1280 × 600 pixels, and below that it shows a short "window too small" message until you enlarge the window or zoom out. Everything runs locally. Nothing is uploaded anywhere (the only downloads are the tools, the starting voice and the Whisper model).

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

Open the page and press **+ New voice**. Give it a short name, choose the folder with your recordings. The starting voice is picked automatically from the pitch of your recordings (it is only the starting point: the result learns your voice from your recordings).

**One step at a time.** The page is a wizard: **Return to …** (it names the previous step) / **Next**, and a **Skip** button on the optional steps (Speakers, Refine, Words). Next unlocks when a step is done, and the message beside it says what is missing. While something runs, that message pulses (for example "Finding the speakers…"), the bar shows real progress with a time estimate, and the footer says what the GPU is busy with. A strip at the top shows where you are: green ✓ done, solid blue current, pulsing red working, dashed dash skipped, grey not reached yet.

The **Show Console** button at the top reveals the console with each job's log. The other extras (voice options such as rename, back up, free up disk space and delete, the version choice when you listen, the starting-voice picker when you create a voice, and free jumping between steps) are always available. If a job fails, everyone gets a plain one-line reason.

Voice names can use capital letters and are case-sensitive (**Joe** and **joe** are two different voices). Rename a voice under the **Voice options** menu; its clips, training and exported files move with it (not while a job is running). At the bottom, **Notify me when a job finishes** plays a chime and shows a desktop notification when a job stops running (the setting is remembered in your browser). To be told even with no browser open, set `notify_url` in the config (a Home Assistant webhook, an ntfy topic, ...): the server then sends one message per finished job. **Other ways to start a voice** (in **+ New voice**, *Start from*): **a prepared dataset** (a folder or .zip with a `metadata.csv` and clips, like the one *Export dataset* makes, or an LJSpeech-style set) skips the speaker and prepare steps; **a backup of a voice** restores everything from a zip made with **Voice options → Back up** (clips, settings, exported files and, if you tick it, the training checkpoints, which are large; the decoded-audio cache and your original recordings are never included).

Every long job has a **Stop** button. **Start over** (bottom of the page) clears a voice's speaker scan, choices and prepared clips so you can begin again; your recordings are never touched and training is only deleted if you choose that too.

| Step | What to do |
|---|---|
| 1. Sources | Choose (Browse…) the folder on your computer that holds your recordings; they are read from there and never copied or changed. Then press **Analyze recordings** (required): it checks every file can be read, measures the total length and converts the recordings the speaker scan will use, so the later steps start faster. Please use only recordings you made yourself or have permission to use. 60 to 120 minutes of clear speech works best; training uses at most 270 minutes. |
| 2. Clean | You choose the cleaning switches, can listen to a random 15-30 s clip cleaned with them (or with each part alone), save the settings, then press Clean. Every recording is cleaned once, whole, into a master copy: [Demucs](https://github.com/facebookresearch/demucs) keeps only the voices (music, effects and background are removed), then rumble and steady hiss are reduced, the voice is brought forward with light compression, and everything is set to one even volume (mono, 22.05 kHz). Every later step (speaker finding, matching, clip cutting, transcripts, training) reads this cleaned master and never the raw files. Your recordings are never changed. Long recordings take a while; there is a progress bar and Stop. |
| 3. Who is speaking? *(optional)* | Only if your recordings have more than one person (a conversation, a call, an interview). Finds each voice (it always listens to all of your recordings), shows samples with the time each comes from, and lets you tick whose voice to train on. Tick ✓ the sample clips that are your speaker. See below. |
| 4. Refine *(optional)* | **Find more like my voice**: looks through the whole recording for more speech that sounds like the clips you ticked, shows how many minutes it found, and lets you tick or cross more and repeat. When a round finds nothing new, it tells you your voice has settled and to go on.  **Keep at most** limits how much matched speech is kept (the closest matches, up to 1.5 times your minutes limit), so Prepare and the later steps have less to process. |
| 5. Build | One click. Splits at pauses, transcribes with Whisper on the GPU, and trims each clip's extra silence (0.15 s kept before the first word and 0.25 s after the last, pauses inside a clip capped at 0.6 s; the untouched originals are kept in `wavs.presilence`). Uses only the people you chose (or the voice you matched). Music and background noise are removed automatically, so only the voices are left (see below).  **Keep the best** (at most 270 minutes, 60 to 120 recommended): if there is more speech than that, Prepare processes an even spread of about 1.5 times the limit, then scores every clip and keeps only the best ones (closest to your speaker, cleanest transcripts). The rest is removed from the dataset, so checking, exporting, backing up and training all stay small. A dataset that is already over the limit can be trimmed with **Keep the best N min now**. Your recordings are never touched. |
| 6. Check the transcripts *(optional)* | **Review** shows one clip at a time (the ones that need attention first): listen with the player, fix the text, then *Keep and next* or *Remove clip*. **List** shows them all, and a Tools panel adds (approve or remove flagged clips, list order). **Check speakers** (under *More tools*) flags clips that sound unlike the rest (another speaker, noise). **The voice learns exactly what is written here, so skipping this may reduce quality** (small slips like "a" for "an" matter little; missing or extra words, wrong names and another person's words matter most). Words from another person can be cut out by deleting them and pressing **Save changes**, below. **Export dataset** (in the Voice options menu) saves the cleaned clips and transcript (22.05 kHz mono wavs + `metadata.csv`) to use anywhere. |
| 7. Train | Quick / Normal / Long, how much speech to use, and **Batch size** (10 to 100% of the largest batch that suits your card; larger trains faster, smaller is gentler on the card; it does not cap video memory, which training reserves most of at any setting). It keeps running if you close the tab; stop and resume any time. The **quality score** (1 to 5, hover it for details) is a predicted listener rating of how natural the test sentences sound; it does not measure how much the voice sounds like you. The progress bar can take 10–20 minutes to update for the first time (it moves when training saves a new checkpoint), so do not stop training until it has updated at least once. |
| 8. Listen and play | Type anything and press play on the player: it renders your line in the trained voice and plays it (new or changed text is rendered first; the same text just replays). **Download wav** saves the last line. **Keep voice in memory** holds the exported voice loaded for near-instant lines, and other programs can use it too (`POST /api/live` with `{voice, text, raw: true}` returns a wav). The **Version** box picks which saved version the voice is exported from. **Compare versions** (a collapsible section) renders a fixed passage with each saved version so you can hear lows, highs, commas, questions and pauses side by side, with a player and **Download wav** on each row and *Use this version* to pick one. The quality score and its tooltip sit under the player. The final button, **Download voice (.zip)**, gives the `.onnx` + `.onnx.json`. |

**Recording tips:** quiet room, same distance from the microphone, no music or fans, no clipping. Read or talk naturally and vary it: questions, short and long sentences, numbers. The model copies how you speak, not just your tone.

### Tuned for Piper

The Piper Mill prepares and trains the way Piper wants it:
- **Audio:** 22.05 kHz mono, clips of 1.5 to 12 seconds cut at natural pauses, every clip levelled to the same loudness (Piper trims leading and trailing silence itself).
- **Text:** transcripts are converted to how the words are *spoken*, because Piper reads text with espeak-ng: "15%" becomes "fifteen percent", "8.30" after "at" becomes "eight thirty", "$3.50" becomes "three dollars and fifty cents", "Dr." becomes "doctor". Repeated-phrase hallucinations and junk are dropped. Check the result in step 4.
- **Training:** fine-tunes from a Piper base voice chosen to match your recordings, so 60 to 120 minutes of speech is plenty. Training uses at most **270 minutes**: if you have more, the cleanest clips that sound most like your speaker are chosen (set the limit under **Use at most** on the Train step; nothing is deleted). Piper holds out 10% of your clips to score progress. If you edit or re-prepare, its cache is refreshed automatically so old audio is never reused.
- **Which version you get:** Piper keeps its best checkpoints by sound quality and by accuracy, plus the latest. The test clip and the downloaded voice default to the **best-sounding** one by quality score, which is usually better than simply the newest. You can pick *Most accurate* or *Latest* in step 6.
- **Disk space:** every checkpoint is about 0.8 GB, so one voice can use up to about 9 GB. *Voice settings → Free up space* keeps only the best and the latest (or `voicestudio clean myvoice`).
- **Output:** `name.onnx` + `name.onnx.json`, the standard Piper voice format.

### Real-world recordings

Prepare, "Who is speaking?" and Build are built for messy folders, not just tidy ones:
- **Long files and big folders:** hours of audio and hundreds of files are fine, with a progress bar and time estimate. No clip is ever longer than 15 seconds (long stretches are cut at the quietest moment).
- **Noisy rooms:** the silence threshold is chosen per recording, and you get a note when a recording is noisy.
- **Any format:** wav, mp3, m4a, flac, ogg, opus, and video files (mp4, mkv, webm). Mono, stereo and surround (surround uses the centre channel, where speech lives) at any sample rate. For videos with several audio tracks, set `audio_language` (for example `"eng"`) in the config; otherwise the first track is used.
- **Files that cannot be used** (corrupt, empty, too short, no speech) are skipped and listed at the end. They never stop the run.

### Recordings with more than one person (optional step 3)

Step 3 uses [pyannote.audio](https://github.com/pyannote/pyannote-audio) speaker diarization (the open `pyannote/speaker-diarization-community-1` model) to find who speaks when. Each voice it finds is listed with how much they speak, when they are **first heard**, and their typical pitch. Sort the list by *Most speech* or *First heard*. Two speakers the model separated inside one recording are never merged; across several recordings, the same person is matched by voice fingerprint (lower `speaker_match_distance` in the config if two people get merged).

Each person has **▶ samples spread through the recording** (the number is the time each comes from), so you hear typical clips and not only the clearest. Tick the person to train on and press **Save my choice**.


**Find more like my voice:** voices can still be mixed up (a person may contain your speaker and similar-sounding others). Press **✓** on sample clips that are your speaker (and **✗** on ones that are not), choose Strict / Normal / Loose, and press *Find more like these*. It compares every stretch of speech in the whole recording with your ticked clips, across all the people found, and shows how many minutes it found plus new samples to check (the ones marked **?** are the least sure). Tick or cross more and run it again to improve it, then choose *Train on the matched voice* or *the ticked people instead*. Two or more ticked clips work best; stretches under one second are skipped because they are too short to recognise reliably.

That model is **gated on Hugging Face**, so once only:
1. Log in at <https://huggingface.co>, open <https://huggingface.co/pyannote/speaker-diarization-community-1> and accept its conditions.
2. Create a read token at <https://huggingface.co/settings/tokens>.
3. Put it in your config as `hf_token = "..."`, or set `HF_TOKEN`, or run `.venv/bin/hf auth login`.

Only use recordings of people who have agreed to it. Telling the tool how many people talk ("auto" by default) improves accuracy.

### Clean audio first (step 2)

The defaults are good for most recordings; the Clean step lets you listen to a sample and change them. Before the speaker scan, before anything is cut or transcribed, each recording is cleaned once, whole, into a master copy (the chain is described in the table above). Cleaning whole recordings avoids the odd noises and dead gaps you get from cleaning short pieces after cutting, and gives the speaker and pause detection clean sound to work with. The result is stored under the voice's `runs/<voice>/master` folder. Older voices that were already past this step keep working from their raw recordings.

### Two voices in one clip: delete their words and save (step 6)

Build cuts at pauses, so a clip can hold two people talking. In *Check the transcripts*, delete the words the wrong person says from the **start or end** of the clip's text, press **Save changes**, which also cuts the edited clips. The Piper Mill listens to each edited clip again with word-by-word timing and cuts the audio down to just the words you kept. Spelling-only fixes are left alone. If words are removed from the **middle**, the clip cannot be cut and you are told to delete it instead. The untrimmed originals are kept in `wavs.untrimmed/`, and training will not start while edited clips are waiting to be reprocessed, so text and audio always match.

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

On the last step, press **Download** for a zip with the two voice files (`name.onnx` and `name.onnx.json`), or use the two download buttons for each file on their own. Put both files, with the same name, in the folder your Piper reads voices from, then restart or reload Piper. They are also kept in the `out` folder of your data folder.

## Settings

Copy `config.example.toml` to `~/.config/voice-studio/config.toml`. Every option is optional (data folder, port, language, Whisper model, batch size, protected voice names).

The page listens on `127.0.0.1` only and has **no login**. Do not expose it to a network you do not trust. It also rejects requests that do not come from the page itself, so other websites cannot drive it.

### Speaker model access (Hugging Face)

The speaker scan, "find more like these", "check clips" and the best-clips ranking in Build use one gated model, `pyannote/speaker-diarization-community-1`. Hugging Face asks you to accept its terms once: create a free account, open the model page and accept the terms, create a read-only token, and put it in `config.toml` as `hf_token` (or run `hf auth login`). The page shows whether access is working and has buttons that open the model page and the token page. A copy of the model (about 40 MB) is stored in the data folder (`models/`) the first time it is needed, so the speaker steps start without asking Hugging Face each time and keep working offline. About once a day, and whenever you press **Check for updates**, the tool asks Hugging Face whether a newer version exists and installs it; offline, it keeps using the copy it has. The copy is made with your own access on your own computer and is never part of this repository. Nothing else the tool downloads is gated.

## Licences and credits

- The Piper Mill itself: MIT (see `LICENSE`).
- The page font, [Open Sans](https://fonts.google.com/specimen/Open+Sans), is embedded in the page under the SIL Open Font License 1.1.
- It installs and drives [piper1-gpl](https://github.com/OHF-Voice/piper1-gpl) (GPL-3.0) as a separate program; it is downloaded by `install.sh` and not included here.
- The speaker model, [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1), is released under CC-BY-4.0 by the pyannote.audio authors. It is downloaded by each user under their own Hugging Face account and is not redistributed here.
- Starting voices come from [rhasspy/piper-checkpoints](https://huggingface.co/datasets/rhasspy/piper-checkpoints). **Each has its own dataset licence, often non-commercial** (for example Ryan is CC BY-NC-SA 4.0). A voice you train from one inherits those limits. The licence is shown when you create a voice and saved next to the file in `base/`.
- Transcription uses OpenAI Whisper (MIT) through Hugging Face `transformers`.
- Removing background noise uses Demucs (MIT) by Meta Research.
- Speaker finding uses pyannote.audio (MIT) with the `speaker-diarization-community-1` model (CC-BY-4.0, gated: accept its conditions on Hugging Face).

## Troubleshooting

- **"CUDA is not available"**: run `nvidia-smi`; reinstall the driver, then `./install.sh` again. Very new GPUs need a PyTorch built for CUDA 12.8 or newer (the default install is).
- **Out of memory while training**: set `batch_size = 8` (or 4) in the config.
- **Whisper is slow or runs out of memory**: set `whisper_model = "openai/whisper-small"`.
- **Transcripts have mistakes**: normal. Fix them in *Check the transcripts* (step 6); it matters more than anything else for quality.
- **Clips are of the wrong person**: pick the right person in step 3, use **Find more like my voice** (step 4), and delete the other person's words and press **Save changes** for clips that mix two people. **Start over** clears it all if you want a clean run.
- **The speaker scan seems stuck**: the bar shows real progress; a long recording can take several minutes. **Stop** cancels it.
- **Voice sounds robotic or mumbles**: more (and cleaner) audio, and more training rounds. Listen to test clips as it trains and stop when it sounds right; more is not always better.

## Developing

`scripts/sanitize_check.sh` scans the repository for private addresses, usernames, tokens and stray models or audio. Run it before publishing.

`scripts/sync.sh` is the publish script: it runs the sanitize check first and stops if anything personal is found, then pushes every commit to a private remote (a full copy of your work, named `gitserver` by default), then pushes `main` to the public remote (`origin`). A local `pre-push` hook can enforce the same check on any push to GitHub.

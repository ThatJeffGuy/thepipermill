# The Piper Mill 2.0: plan (not implemented yet)

Status: design only. Nothing here exists in the code yet.

## Idea
After **Create voice**, the wizard becomes a short tree. The user picks how the voice will be made:

```
Create voice
  └─ Prepare the audio (clean it, check it)          <- shared by both paths
        ├─ A. Classic: train on your recordings      (what 1.x does, minus multi-speaker)
        └─ B. Fast: clone the voice, then fine-tune  (new)
```

## Shared step: Clean the audio (new, before any processing)
- Checkboxes: remove music/background (Demucs, on by default), denoise (off by default), more later.
- Preview 3 to 5 varied clips (quiet, loud, long): original and cleaned side by side, with **Approve** or **Deny** on each.
- Deny: change the boxes and preview again. Nothing is processed until the user approves.
- Note shown to the user: heavy denoising can smooth out a gravelly or breathy voice.

## Path A: Classic training
Fine-tune a Piper base voice directly on the user's own clips. Same as 1.x, but single speaker only (see "Removed" below).

| Pros | Cons |
|---|---|
| Uses only real recordings; nothing synthetic | Needs a lot of clean speech (60 to 120 minutes is recommended) |
| Keeps real delivery and character in the data | Training takes hours on the GPU |
| Simple to explain | Noisy or mixed-speaker audio gives a poor voice; overfits easily on small sets |

## Path B: Fast: clone, then fine-tune
1. Take one clean reference clip (about 7 to 12 seconds) with its transcript.
2. A zero-shot cloning model (F5-TTS) speaks a few hundred varied sentences in that voice.
3. Filter the results (Whisper word check and speaker similarity), keep 30 to 60 minutes.
4. A short Piper fine-tune on that clean synthetic set produces a normal fast Piper voice (.onnx).

| Pros | Cons |
|---|---|
| Needs only seconds of audio, not hours | The model learns the clone, so delivery is only as expressive as the cloner |
| Clean, consistent training data; no diarization or transcript editing | Text matters: neutral sentences sound flat, so the sentence set should match the speaker's register |
| Whole process is much shorter | Extra install (the cloner, a few GB) and its licence terms (F5-TTS weights are non-commercial) |
| Result is a normal Piper voice, fast at run time | Quality is bounded by the reference clip; a few clips can drift and must be filtered |

Open questions for Path B:
- Sentence sources: an LLM writing lines in the speaker's register from a few example transcripts, or the user's own text.
- One reference or several (for example a calm and a bright clip)? Early tests: references change pitch and style, so the user should pick by ear.
- Optionally mix some real clips into the final training set.
- Quality gates before training (naturalness score, word accuracy, speaker similarity).

## Removed in 2.0
- Multi-speaker scan, diarization, "find more like my voice", and the staged training that went with them. Input is one speaker per dataset.

## Release checklist (when work starts)
- Clean-the-audio step with preview and approve/deny.
- Tree screen after Create voice with the two paths and the pros/cons above.
- Path B pipeline as a separate job type with its own console log and a status bar like the other jobs.
- Remove the multi-speaker steps and update the README, the step strip and the tests.

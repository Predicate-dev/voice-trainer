# Voice Trainer - Speech Coach PoC

A Python command-line speech coaching system that provides real-time feedback on speaking skills. This proof-of-concept uses speech recognition, audio analysis, and text-to-speech to help users improve their speaking pace, volume, and tone variation.

## What's new (recent additions)

- Desktop GUI (PyQt5) with start/stop controls, session review, and TTS preview.
- Realtime sparkline plots (pyqtgraph) for key DSP metrics: F0, Jitter, HNR, and an optional model confidence score.
- Vosk integration for fast offline realtime recognition plus a Vosk fallback for session transcripts when Whisper/ffmpeg are not available.
- Whisper integration for higher-quality end-of-session transcription when `ffmpeg` is installed; the app will automatically try Whisper first and fall back to Vosk if Whisper cannot load audio.
# Voice Trainer — Speech Coach

Short, opinionated README focused on practical steps and clarity. If you want screenshots or an expanded developer guide I can add them.

What this repo contains
- A small Python-based speech coaching PoC with both a CLI (`main.py`) and a desktop GUI (`speech_coach_app.py`).
- Core analysis and orchestration live in `speech_coach.py`.

Goals
- Give realtime feedback (WPM, RMS volume, pitch/F0 variation).
- Use offline realtime ASR (Vosk) for live feedback and Whisper for higher-quality post-session transcripts when available.
- Provide advanced DSP metrics and an optional MFCC→PyTorch inference hook for low-latency scoring.

---

## Quick start (3 commands)

Clone, venv, install:

```bash
git clone https://github.com/Predicate-dev/voice-trainer.git
cd voice-trainer
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run the GUI:

```bash
python speech_coach_app.py
```

Or run a quick CLI session/demo:

```bash
python demo.py
python test_speech_coach.py   # simulation without mic
python main.py               # interactive CLI (requires mic)
```

---

## Installation details

Core Python deps are in `requirements.txt`. For optional features install:

```bash
pip install librosa scipy torch vosk pyqtgraph
```

System packages you may need:
- PortAudio (macOS: `brew install portaudio`, Ubuntu: `sudo apt-get install portaudio19-dev`)
- ffmpeg (required by Whisper): `brew install ffmpeg` or `sudo apt-get install ffmpeg`

Notes:
- Whisper requires `ffmpeg` on PATH to load audio. If missing, the app falls back to Vosk for end-of-session transcription.
- PyTorch and a model file are optional — the app works without them.

---

## Main features (short)

- Realtime GUI with Start/Stop, TTS preview, and live DSP metrics.
- Vosk for offline realtime recognition (word timestamps, partial results).
- Whisper (optional) for higher-quality end-of-session transcripts.
- DSP: F0 (autocorr), jitter/shimmer approximations, HNR, spectral centroid/flatness, MFCCs.
- Optional PyTorch inference worker on MFCC windows (put model at `models/mfcc_model.pt`).
- Adaptive baselines saved to `user_baseline.json`.

---

## Files & where to put extra assets

- Vosk model (user): `vosk-model/vosk-model-small-en-us-0.15/` — download and unzip from the Vosk site.
- Optional MFCC PyTorch model: `models/mfcc_model.pt` (user-provided).
- Baselines: `user_baseline.json` — created/updated automatically.

---

## Troubleshooting (practical fixes)

1) No transcript or "ffmpeg" error

- Install ffmpeg (see above) to enable Whisper. If you can't or don't want to install ffmpeg the app will fall back to Vosk for a session transcript.

2) Native audio crashes / exit code 134

- Reinstall PortAudio and rebuild PyAudio inside the venv:

```bash
brew install portaudio     # macOS
pip install --force-reinstall --no-binary :all: pyaudio
```

3) GUI plots not showing or broken

- Ensure `pyqtgraph` is installed. If plotting still fails, run `python speech_coach_app.py` from a terminal and inspect console logs.

4) Model doesn't load

- Check that `torch` is installed and `models/mfcc_model.pt` exists. The app logs any model load error and continues in heuristic-only mode.

5) Baseline feels wrong

- Delete `user_baseline.json` to reset learned baselines.

---

## Developer notes

- Keep optional imports behind guards (librosa, torch, vosk). The application must run degraded but usable if optional libs are not present.
- DSP helpers live in `speech_coach.py` — thread-safety is important for shared buffers and the optional inference worker.
- Tests: `test_speech_coach.py` is a lightweight simulation harness.

---

If you'd like, I can implement one of these follow-ups now:
- Add a "Retry Whisper" button to the GUI (run Whisper post-session once ffmpeg is installed).
- Add per-plot toggles and autoscaling for the pyqtgraph plots.
- Add a small `setup-macos.sh` script that installs Homebrew formulae and bootstraps the venv.

Tell me which and I'll implement it.



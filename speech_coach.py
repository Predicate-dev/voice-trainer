#!/usr/bin/env python3
import os
os.environ["NUMBA_THREADING_LAYER"] = "tbb"
"""
Speech Coach PoC - Real-time speech analysis and feedback system.

Features:
- Pacing: Measures WPM and provides feedback if speech is too fast
- Volume: Calculates RMS and provides feedback if speech is too quiet
- Tone: Analyzes pitch variation and provides feedback if speech is monotonous
"""

import speech_recognition as sr
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("⚠️  PyAudio not available. Volume and tone analysis will be limited.")

import pyttsx3
import numpy as np
import threading
import time
import sys
from collections import deque
from typing import Optional, List, Tuple
import soundfile as sf
import pyaudio


import threading
import sys
import time
import select

import tempfile
import os
try:
    import librosa
    LIBROSA_AVAILABLE = True
except Exception:
    LIBROSA_AVAILABLE = False
    librosa = None

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None

import scipy.signal as signal


class SpeechCoach:
    # Baseline and adaptive goals
    baseline_file = "user_baseline.json"

    def compute_baseline(self):
        """Compute baseline metrics from the last session."""
        import numpy as np
        baseline = {
            "avg_wpm": np.mean([w for _, w in self.word_timestamps]) if self.word_timestamps else 0,
            "avg_volume": np.mean(self.rms_values) if self.rms_values else 0,
            "avg_pitch_var": np.std(self.pitch_history) if self.pitch_history else 0,
            "filler_rate": self.filler_word_total / max(1, self.total_words),
            "session_count": 1
        }
        return baseline

    def save_baseline(self, baseline):
        import json
        # If file exists, average with previous baseline
        try:
            with open(self.baseline_file, "r") as f:
                prev = json.load(f)
            # Running average for each metric
            for k in ["avg_wpm", "avg_volume", "avg_pitch_var", "filler_rate"]:
                prev[k] = (prev[k] * prev["session_count"] + baseline[k]) / (prev["session_count"] + 1)
            prev["session_count"] += 1
            baseline = prev
        except Exception:
            pass
        # Ensure all values are plain Python types for JSON
        def _to_json_safe(x):
            try:
                import numpy as _np
                if isinstance(x, _np.floating):
                    return float(x)
                if isinstance(x, _np.integer):
                    return int(x)
            except Exception:
                pass
            if isinstance(x, float):
                return float(x)
            if isinstance(x, int):
                return int(x)
            if isinstance(x, dict):
                return {k: _to_json_safe(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [_to_json_safe(v) for v in x]
            return x

        safe_baseline = _to_json_safe(baseline)
        with open(self.baseline_file, "w") as f:
            json.dump(safe_baseline, f, indent=2)

    def load_baseline(self):
        import json
        try:
            with open(self.baseline_file, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def update_goals_from_baseline(self):
        """Set adaptive thresholds based on user baseline."""
        baseline = self.load_baseline()
        if not baseline:
            return
        # Example: set goals to +10% of baseline (or other heuristics)
        self.wpm_threshold = max(100, baseline["avg_wpm"] * 1.1)
        self.volume_threshold = max(0.005, baseline["avg_volume"] * 1.1)
        self.pitch_variation_threshold = max(0.05, baseline["avg_pitch_var"] * 1.2)
        # Optionally, set other goals (e.g., reduce filler rate)
        self.filler_rate_goal = max(0.01, baseline["filler_rate"] * 0.8)

    

    """Real-time speech analysis and coaching system with start/pause/stop triggers, session review, and two modes."""

    def __init__(self, mode="freestyle", reference_text=None):
        # Mode and reference
        self.mode = mode
        self.reference_text = reference_text
        self.transcript = []  # For speech mode: store recognized text
        self.audio_record_path = None  # Path to temp WAV file for Whisper
        # Speech recognition setup
        self.recognizer = sr.Recognizer()
        try:
            self.microphone = sr.Microphone()
            self.microphone_available = True
        except Exception as e:
            print(f"  Microphone not available: {e}")
            self.microphone_available = False

        # Text-to-speech setup
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)  # Slower speech for feedback
            self.tts_available = True
        except Exception as e:
            print(f"⚠️  TTS not available: {e}")
            self.tts_available = False

        # Audio analysis parameters
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.channels = 1

        # Thresholds for feedback
        self.wpm_threshold = 180  # Words per minute threshold
        self.volume_threshold = 0.01  # RMS threshold for volume
        self.pitch_variation_threshold = 0.2  # Minimum pitch variation

        # Analysis state
        self.audio_buffer = deque(maxlen=50)  # Keep last 50 audio chunks
        self.word_timestamps = deque(maxlen=20)  # Keep last 20 word timestamps
        self.pitch_values = deque(maxlen=30)  # Keep last 30 pitch measurements

        # Control flags
        self.running = False
        self.paused = False
        self.session_active = False
        self.audio_thread = None
        self.analysis_thread = None
        self.key_thread = None

        # Session stats
        self.session_start_time = None
        self.session_end_time = None
        self.total_words = 0
        self.max_wpm = 0
        self.min_wpm = float('inf')
        self.low_volume_count = 0
        self.loud_volume_count = 0
        self.monotone_count = 0
        self.rms_values = []  # Store all RMS values for session
        self.pitch_history = []  # Store all pitch (ZCR) values for session
        # DSP histories
        self.f0_history = []
        self.jitter_history = []
        self.shimmer_history = []
        self.hnr_history = []
        self.mfcc_buffer = deque(maxlen=50)
        self.realtime_model_score = None
        # Spectral / FFT-based clarity metrics
        self.spectral_centroid_history = []
        self.spectral_flatness_history = []
        self.harmonic_energy_ratio_history = []
        # Locks for thread-safe access
        self._mfcc_lock = threading.Lock()
        self._score_lock = threading.Lock()

        # Try to load a PyTorch model for MFCC time-series analysis if present
        self.mfcc_model = None
        self.mfcc_model_path = os.path.join("models", "mfcc_model.pt")
        # Runtime toggles (can be controlled by GUI to reduce native/audio/model risk)
        self.enable_dsp = True
        self.enable_inference = True
        if TORCH_AVAILABLE and os.path.exists(self.mfcc_model_path):
            try:
                # Prefer torch.jit for faster startup when available
                try:
                    self.mfcc_model = torch.jit.load(self.mfcc_model_path)
                    print(f"Loaded MFCC model from {self.mfcc_model_path}")
                except Exception:
                    self.mfcc_model = torch.load(self.mfcc_model_path)
                    print(f"Loaded MFCC model (torch.load) from {self.mfcc_model_path}")
            except Exception as e:
                print(f"⚠️  Failed to load MFCC model: {e}")
        # Start lightweight inference worker for sub-100ms feedback if model loaded
        self.inference_thread = None
        self.inference_active = False
        # Only start if model is present and inference is enabled
        if self.mfcc_model is not None and self.enable_inference:
            self.inference_active = True
            self.inference_thread = threading.Thread(target=self._inference_worker)
            self.inference_thread.daemon = True
            self.inference_thread.start()
        self.loud_threshold = 0.2  # RMS threshold for too loud (customizable)

        # Feedback options
        self.enabled_feedback = {"pacing", "volume", "tone", "filler", "pronunciation", "emotion", "visual"}
        self.language = "en"
        # Feedback cooldowns (prevent spam)
        self.last_pacing_feedback = 0
        self.last_volume_feedback = 0
        self.last_tone_feedback = 0
        self.feedback_cooldown = 5  # seconds

        # Rhythm and pausing analysis
        self.pause_durations = []  # List of detected pauses (seconds)
        self.long_pause_count = 0
        self.irregular_rhythm_count = 0
        self.pause_threshold = 1.2  # seconds (customizable)
        self.irregular_rhythm_threshold = 0.7  # stddev of pause durations (seconds)

        # Filler word detection
        self.filler_words = ["um", "uh", "like", "you know", "so", "actually", "basically", "literally", "right", "okay", "well"]
        self.filler_word_counts = {}
        self.filler_word_total = 0

        # Emotion and expressiveness analysis
        self.emotion_score = 0
        self.emotion_label = ""
        
    def start(self, gui_mode=False):
        # Load adaptive goals before session
        self.update_goals_from_baseline()
        """Start the speech coaching session. If gui_mode, start immediately and skip keyboard triggers."""
        print("🎤 Speech Coach Ready!")
        if not self.microphone_available:
            # Fallback: if PyAudio is available, continue using _audio_capture_loop instead
            if PYAUDIO_AVAILABLE:
                print("⚠️  speech_recognition Microphone not available; falling back to PyAudio capture.")
            else:
                print(" Cannot start: No microphone available and PyAudio not available.")
                return
        else:
            # Only calibrate if we have an sr.Microphone source
            try:
                print("Calibrating microphone...")
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=2)
                print(" Calibration complete!")
            except Exception as e:
                print(f"Microphone calibration failed, continuing if PyAudio is available: {e}")
        self.running = True
        if gui_mode:
            self.paused = False
            self.session_active = True
            self.session_start_time = time.time()
        else:
            self.paused = True
            self.session_active = False
        if not gui_mode:
            print("Controls: [r] Start  [p] Pause/Resume  [s] Stop  [Ctrl+C] Quit")
            print("=" * 50)
            self.key_thread = threading.Thread(target=self._key_listener)
            self.key_thread.daemon = True
            self.key_thread.start()
        if self.mode == "speech":
            # Prepare temp WAV file for recording
            self.audio_record_path = tempfile.mktemp(suffix=".wav")
            self.audio_thread = threading.Thread(target=self._record_audio_wav)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            print(f"[DEBUG] audio_thread started for speech mode, recording to {self.audio_record_path}")
        elif PYAUDIO_AVAILABLE:
            self.audio_thread = threading.Thread(target=self._audio_capture_loop)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            print("[DEBUG] audio_thread started using PyAudio capture")
        self.analysis_thread = threading.Thread(target=self._analysis_loop)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        print("[DEBUG] analysis_thread started")
        self._speech_recognition_loop()

    def _record_audio_wav(self):
        """Record the session to a WAV file for Whisper transcription (speech mode only)."""
        
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        frames = []
        print("[DEBUG] Recording WAV thread active")
        while self.running:
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
            if len(frames) % 50 == 0:
                print(f"[DEBUG] recorded frames: {len(frames)}")
        stream.stop_stream()
        stream.close()
        p.terminate()
        # Save to WAV
        audio_bytes = b"".join(frames)
        import numpy as np
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        sf.write(self.audio_record_path, audio_np, 16000)
    
    def stop(self):
        """Stop the speech coaching session and print review."""
        print("\n Stopping Speech Coach...")
        self.running = False
        self.paused = True
        self.session_active = False
        if self.audio_thread:
            self.audio_thread.join(timeout=1)
        if self.analysis_thread:
            self.analysis_thread.join(timeout=1)
        import threading
        if self.key_thread and threading.current_thread() != self.key_thread:
            self.key_thread.join(timeout=1)
        self.session_end_time = time.time()
        print("✅ Speech Coach stopped.")
        # After session, compute and save baseline
        baseline = self.compute_baseline()
        self.save_baseline(baseline)
        # If in speech mode, transcribe with Whisper before review
        if self.mode == "speech" and self.audio_record_path:
            self._whisper_transcribe()
        self._print_session_review()

    def _whisper_transcribe(self):
        """Transcribe the recorded WAV file using OpenAI Whisper."""
        if not self.audio_record_path:
            print("No audio file to transcribe.")
            self.transcript = []
            self.transcript_text = ""
            return
        print("\n Transcribing with Whisper (this may take a moment)...")
        import whisper
        try:
            model = whisper.load_model("base")
            result = model.transcribe(self.audio_record_path, word_timestamps=True, verbose=False)
            # result["text"] is always a string
            self.transcript_text = str(result["text"]) if result["text"] is not None else ""
            self.transcript = self.transcript_text.split() if self.transcript_text else []
        except FileNotFoundError as e:
            # Often caused by missing ffmpeg executable used by whisper's audio loader
            print(f"Whisper transcription skipped: external dependency missing: {e}")
            # Try to fall back to Vosk (offline) if available and model exists
            try:
                from vosk import Model as VoskModel, KaldiRecognizer
                import wave
                import json
                model_path = "vosk-model/vosk-model-small-en-us-0.15"
                if os.path.exists(model_path):
                    print("Attempting Vosk fallback transcription...")
                    wf = wave.open(self.audio_record_path, "rb")
                    if wf.getnchannels() != 1 or wf.getframerate() != 16000:
                        # We expect 16k mono WAV; try to read and convert using soundfile if available
                        try:
                            import soundfile as sf
                            data, sr = sf.read(self.audio_record_path)
                            # Convert to mono if necessary
                            if len(data.shape) > 1:
                                data = data.mean(axis=1)
                            # Resample if needed
                            if sr != 16000:
                                try:
                                    import librosa
                                    data = librosa.resample(data.astype('float32'), orig_sr=sr, target_sr=16000)
                                except Exception:
                                    pass
                            # Write a temp 16k mono WAV for Vosk
                            import tempfile
                            temp_wav = tempfile.mktemp(suffix=".wav")
                            sf.write(temp_wav, data, 16000)
                            wf = wave.open(temp_wav, 'rb')
                        except Exception:
                            pass
                    vosk_model = VoskModel(model_path)
                    rec = KaldiRecognizer(vosk_model, 16000)
                    rec.SetWords(True)
                    results = []
                    while True:
                        data = wf.readframes(4000)
                        if len(data) == 0:
                            break
                        if rec.AcceptWaveform(data):
                            res = json.loads(rec.Result())
                            results.append(res.get('text', ''))
                    # final
                    try:
                        final_res = json.loads(rec.FinalResult())
                        results.append(final_res.get('text', ''))
                    except Exception:
                        pass
                    transcript_text = ' '.join([r for r in results if r])
                    self.transcript_text = transcript_text
                    self.transcript = transcript_text.split() if transcript_text else []
                    print("Vosk fallback transcript:", self.transcript_text)
                    try:
                        wf.close()
                    except Exception:
                        pass
                    return
                else:
                    print("Vosk model not found for fallback.")
            except Exception as e2:
                print(f"Vosk fallback failed: {e2}")
            # If Vosk fallback didn't run, set empty transcript
            self.transcript = []
            self.transcript_text = ""
            return
        except Exception as e:
            print(f"Whisper transcription failed: {e}")
            self.transcript = []
            self.transcript_text = ""
            return
        # Pronunciation feedback: collect low-confidence words
        self.mispronounced_words = []
        if "segments" in result:
            for seg in result["segments"]:
                if isinstance(seg, dict):
                    words = seg.get("words", [])
                    if isinstance(words, list):
                        for w in words:
                            # Defensive: Only process if w is a dict (not str)
                            if isinstance(w, dict):
                                # Whisper confidence is 0-1, flag low confidence
                                if w.get("confidence", 1.0) < 0.85:
                                    self.mispronounced_words.append(w.get("word", ""))
        print(" Whisper transcript:")
        print(self.transcript_text)
        # Filler word detection on transcript
        self._detect_filler_words(self.transcript_text)
        # Clean up temp file
        if self.audio_record_path:
            try:
                os.remove(self.audio_record_path)
            except Exception:
                pass

    def _detect_filler_words(self, text):
        """Detect and count filler words in the given text."""
        import re
        self.filler_word_counts = {}
        self.filler_word_total = 0
        text_lower = text.lower()
        for word in self.filler_words:
            # Use word boundaries for single words, substring for phrases
            if " " in word:
                count = text_lower.count(word)
            else:
                count = len(re.findall(rf'\b{re.escape(word)}\b', text_lower))
            if count > 0:
                self.filler_word_counts[word] = count
                self.filler_word_total += count

    def _key_listener(self):
        """Listen for keyboard input to control start/pause/stop."""
        while self.running:
            if sys.platform == 'win32':
                import msvcrt
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8').lower()
                    self._handle_key(key)
            else:
                # Unix: use select for non-blocking stdin
                dr, _, _ = select.select([sys.stdin], [], [], 0.1)
                if dr:
                    key = sys.stdin.read(1).lower()
                    self._handle_key(key)
            time.sleep(0.05)

    def _handle_key(self, key):
        if key == 'r':
            if not self.session_active:
                print(" Session started!")
                self.session_active = True
                self.paused = False
                self.session_start_time = time.time()
            elif self.paused:
                print(" Resumed.")
                self.paused = False
        elif key == 'p':
            if self.session_active and not self.paused:
                print(" Paused.")
                self.paused = True
        elif key == 's':
            print(" Stop key pressed.")
            self.stop()
    
    def _audio_capture_loop(self):
        """Capture audio for real-time analysis."""
        if not PYAUDIO_AVAILABLE:
            return
        import pyaudio as _pyaudio
        p = _pyaudio.PyAudio()
        stream = None
        try:
            stream = p.open(
                format=_pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            print("[DEBUG] PyAudio stream opened for capture")
            while self.running:
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.float32)
                    self.audio_buffer.append(audio_data)
                    if len(self.audio_buffer) % 10 == 0:
                        print(f"[DEBUG] audio_buffer length: {len(self.audio_buffer)}")
                except Exception as e:
                    print(f" Audio capture error: {e}")
                    break
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            p.terminate()
    
    def _speech_recognition_loop(self):
        """Continuous speech recognition for WPM calculation using Vosk."""
        try:
            from vosk import Model, KaldiRecognizer
            import pyaudio
            import json
        except ImportError:
            print("Vosk is not installed. Please install it with pip install vosk.")
            return
        model_path = "vosk-model/vosk-model-small-en-us-0.15"
        if not os.path.exists(model_path):
            print(f"Vosk model not found at {model_path}. Please download and unzip the model.")
            return
        model = Model(model_path)
        recognizer = KaldiRecognizer(model, self.sample_rate)
        recognizer.SetWords(True)
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, input=True, frames_per_buffer=self.chunk_size)
        stream.start_stream()
        buffer = b''
        partial_transcript = []
        while self.running:
            if not self.session_active or self.paused:
                time.sleep(0.1)
                continue
            try:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                # Feed audio to analysis buffer for volume/pitch
                audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                self.audio_buffer.append(audio_np)
                buffer += data
                # Process partial results for live transcript and word count
                partial = recognizer.PartialResult()
                partial_json = json.loads(partial)
                partial_text = partial_json.get('partial', '').strip()
                if partial_text:
                    words = partial_text.split()
                    # Only add new words
                    new_words = [w for w in words if w not in partial_transcript]
                    if new_words:
                        partial_transcript.extend(new_words)
                        word_count = len(new_words)
                        current_time = time.time()
                        self.word_timestamps.append((current_time, word_count))
                        self.total_words += word_count
                        if self.mode == "freestyle":
                            print(f"📝 Partial: {' '.join(new_words)}")
                        elif self.mode == "speech":
                            self.transcript.extend(new_words)
                # Process final results for completed phrases
                if recognizer.AcceptWaveform(data):
                    result = recognizer.Result()
                    result_json = json.loads(result)
                    text = result_json.get('text', '').strip()
                    if text:
                        words = text.split()
                        word_count = len(words)
                        current_time = time.time()
                        self.word_timestamps.append((current_time, word_count))
                        self.total_words += word_count
                        # Rhythm and pausing analysis: detect pauses between utterances
                        if hasattr(self, "_last_word_time"):
                            pause = current_time - self._last_word_time
                            self.pause_durations.append(pause)
                            if pause > self.pause_threshold:
                                self.long_pause_count += 1
                        self._last_word_time = current_time
                        if self.mode == "freestyle":
                            print(f"📝 Recognized: {text}")
                        elif self.mode == "speech":
                            self.transcript.extend(words)
            except Exception as e:
                print(f" Vosk recognition loop error: {e}")
                time.sleep(0.1)
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    def _analysis_loop(self):
        """Continuous analysis of speech metrics."""
        while self.running:
            if not self.session_active or self.paused:
                time.sleep(0.1)
                continue
            try:
                current_time = time.time()
                # Analyze volume (only if PyAudio available)
                if PYAUDIO_AVAILABLE:
                    self._analyze_volume()
                # Analyze pacing (WPM)
                self._analyze_pacing(current_time)
                # Analyze tone (pitch variation) (only if PyAudio available)
                if PYAUDIO_AVAILABLE:
                    self._analyze_tone()
                # Advanced DSP analysis (F0, Jitter, Shimmer, HNR, MFCCs)
                if PYAUDIO_AVAILABLE and getattr(self, 'enable_dsp', True):
                    self._analyze_dsp()
                # Analyze emotion/expressiveness (pitch/volume stats)
                self._analyze_emotion()
                # Print live metrics
                self._print_live_metrics()
                time.sleep(1)
            except Exception as e:
                print(f" Analysis error: {e}")
                time.sleep(1)

    # --- DSP Utilities & Analysis ---
    def _analyze_dsp(self):
        """Compute F0, jitter, shimmer, HNR, and MFCCs from recent audio and optionally run model inference."""
        if not self.audio_buffer:
            return
        # Use last ~0.2-0.5s of audio for stable DSP
        sr = self.sample_rate
        needed = int(0.25 * sr)
        recent = np.concatenate(list(self.audio_buffer)[-10:])
        if len(recent) < 1024:
            return
        # Take last 'needed' samples
        audio = recent[-needed:]
        # F0 estimation (autocorrelation)
        f0 = self._estimate_f0_autocorr(audio, sr)
        if f0 is not None:
            self.f0_history.append(f0)
        # HNR
        hnr = self._estimate_hnr(audio)
        if hnr is not None:
            self.hnr_history.append(hnr)
            # Provide feedback if HNR too low (noisy/hoarse)
            if hnr < 15 and time.time() - self.last_tone_feedback > self.feedback_cooldown:
                self._provide_feedback("🔎 Your voice has a hoarse/noisy quality (low HNR). Consider hydrating or warm-ups.")
                self.last_tone_feedback = time.time()
        # FFT-based spectral features for clarity
        try:
            sc = self._spectral_centroid(audio, sr)
            if sc is not None:
                self.spectral_centroid_history.append(sc)
            sfm = self._spectral_flatness(audio)
            if sfm is not None:
                self.spectral_flatness_history.append(sfm)
            her = self._harmonic_energy_ratio(audio, sr)
            if her is not None:
                self.harmonic_energy_ratio_history.append(her)
                # Feedback if harmonic energy is low (indicates breathiness/noise)
                if her < 0.2 and time.time() - self.last_tone_feedback > self.feedback_cooldown:
                    self._provide_feedback("🔊 Low harmonic energy detected — speak more clearly and reduce breathiness.")
                    self.last_tone_feedback = time.time()
        except Exception as e:
            print(f"DSP spectral feature error: {e}")

        # Jitter & Shimmer (approximate from F0 and amplitude)
        if len(self.f0_history) >= 3:
            jitter = self._estimate_jitter(self.f0_history[-20:])
            shimmer = self._estimate_shimmer(audio)
            if jitter is not None:
                self.jitter_history.append(jitter)
            if shimmer is not None:
                self.shimmer_history.append(shimmer)
        # MFCC extraction
        mfcc = self._compute_mfcc(audio, sr)
        if mfcc is not None:
            with self._mfcc_lock:
                self.mfcc_buffer.append(mfcc)
            # Run model inference if available and enough frames
                # inference worker will pick this up
                pass

    def _estimate_f0_autocorr(self, audio: np.ndarray, sr: int, fmin=50, fmax=500):
        """Estimate F0 using autocorrelation method."""
        # Pre-emphasis
        audio = signal.lfilter([1, -0.97], [1], audio)
        # Windowing
        win = np.hanning(len(audio))
        x = audio * win
        # Autocorrelation
        corr = np.correlate(x, x, mode='full')
        corr = corr[len(corr)//2:]
        if np.all(corr == 0):
            return None
        # Find peak in lag range
        min_lag = int(sr / fmax)
        max_lag = int(sr / fmin)
        if max_lag <= min_lag:
            return None
        segment = corr[min_lag:max_lag]
        peak = np.argmax(segment) + min_lag
        if corr[peak] <= 0:
            return None
        f0 = sr / peak
        return float(f0)

    def _estimate_hnr(self, audio: np.ndarray):
        """Estimate Harmonics-to-Noise Ratio (HNR) via autocorrelation peak strength."""
        try:
            audio = audio - np.mean(audio)
            corr = np.correlate(audio, audio, mode='full')
            corr = corr[len(corr)//2:]
            if len(corr) < 2:
                return None
            # Peak at lag 0 is energy; search next peak
            peak0 = corr[0]
            # find maximum autocorrelation after lag 0 within reasonable range
            search = corr[1: int(len(corr)*0.5)]
            peak1 = np.max(search) if len(search) > 0 else 0.0
            # HNR in dB
            if peak0 - peak1 <= 0:
                return 0.0
            hnr = 10 * np.log10((peak1) / (peak0 - peak1 + 1e-8) + 1e-8)
            # Bound
            return float(hnr)
        except Exception:
            return None

    def _estimate_jitter(self, f0_list: List[float]):
        """Approximate jitter as relative average absolute period differences."""
        try:
            periods = [1.0 / f for f in f0_list if f > 0]
            if len(periods) < 2:
                return None
            diffs = np.abs(np.diff(periods))
            jitter = np.mean(diffs) / np.mean(periods)
            return float(jitter)
        except Exception:
            return None

    def _estimate_shimmer(self, audio: np.ndarray):
        """Approximate shimmer as cycle-to-cycle amplitude variation using Hilbert envelope."""
        try:
            analytic = signal.hilbert(audio)
            env = np.abs(analytic)
            # Split into short frames and compute per-frame peak
            frame_len = int(0.02 * self.sample_rate)
            if frame_len < 1:
                return None
            peaks = [np.max(env[i:i+frame_len]) for i in range(0, len(env)-frame_len, frame_len)]
            if len(peaks) < 2:
                return None
            peaks = np.array(peaks)
            diffs = np.abs(np.diff(peaks))
            shimmer = np.mean(diffs) / (np.mean(peaks) + 1e-8)
            return float(shimmer)
        except Exception:
            return None

    def _compute_mfcc(self, audio: np.ndarray, sr: int, n_mfcc=13):
        if not LIBROSA_AVAILABLE or librosa is None:
            return None
        try:
            # librosa expects float32
            audio_f = audio.astype('float32')
            mfcc = librosa.feature.mfcc(y=audio_f, sr=sr, n_mfcc=n_mfcc)
            # Return as (n_mfcc, T)
            return mfcc.astype(np.float32)
        except Exception as e:
            print(f"MFCC extraction error: {e}")
            return None

    def _run_mfcc_model_inference(self):
        """Run the loaded PyTorch model on the latest MFCC frames and return a score (0-1)."""
        if not TORCH_AVAILABLE or self.mfcc_model is None or torch is None:
            return None
        # Stack last N mfcc frames into a tensor
        try:
            # Each entry in mfcc_buffer is (n_mfcc, T_frame). We can concatenate along time
            mfccs = list(self.mfcc_buffer)
            if not mfccs:
                return None
            concat = np.concatenate(mfccs, axis=1)  # (n_mfcc, total_T)
            # Normalize
            concat = (concat - np.mean(concat)) / (np.std(concat) + 1e-8)
            # Convert to tensor shape (1, 1, n_mfcc, T)
            try:
                tensor = torch.from_numpy(concat).unsqueeze(0).unsqueeze(0).float()
                with torch.no_grad():
                    out = self.mfcc_model(tensor)
            except Exception as e:
                print(f"Error preparing tensor or running model: {e}")
                return None
                # Expect model to return probability-like scalar
                if isinstance(out, torch.Tensor):
                    score = torch.sigmoid(out).cpu().numpy()
                    # Flatten to scalar if needed
                    if score.size == 1:
                        return float(score.item())
                    return float(score.flatten()[0])
                else:
                    return float(out)
        except Exception as e:
            print(f"Error running mfcc model inference: {e}")
            return None

    def _analyze_emotion(self):
        """Estimate emotional tone using pitch and volume stats."""
        # Only analyze if enough data
        if not self.rms_values or not self.pitch_history:
            self.emotion_score = 0
            self.emotion_label = "Not enough data"
            return
        import numpy as np
        pitch_sd = np.std(self.pitch_history)
        volume_sd = np.std(self.rms_values)
        avg_pitch = np.mean(self.pitch_history)
        avg_volume = np.mean(self.rms_values)
        # Simple scoring: more variation = more expressive
        expressiveness = pitch_sd + volume_sd
        self.emotion_score = expressiveness
        # Heuristic emotion label
        if expressiveness > 0.15:
            if avg_pitch > 0.1 and avg_volume > 0.05:
                self.emotion_label = "Excited/Expressive"
            else:
                self.emotion_label = "Expressive"
        elif expressiveness > 0.08:
            self.emotion_label = "Conversational"
        elif expressiveness > 0.04:
            self.emotion_label = "Calm/Flat"
        else:
            self.emotion_label = "Monotone/Low energy"

    # --- Additional DSP helpers ---
    def _inference_worker(self):
        """Continuously run fast inference on recent MFCC frames for low-latency feedback."""
        if not TORCH_AVAILABLE or self.mfcc_model is None:
            return
        import time
        while self.inference_active:
            try:
                with self._mfcc_lock:
                    if len(self.mfcc_buffer) < 3:
                        pass
                    else:
                        score = self._run_mfcc_model_inference()
                        with self._score_lock:
                            self.realtime_model_score = score
                        # If model signals poor clarity, provide feedback
                        if score is not None and score < 0.35 and time.time() - self.last_tone_feedback > self.feedback_cooldown:
                            self._provide_feedback("🧭 Quick check: try clearer articulation and controlled pacing.")
                            self.last_tone_feedback = time.time()
            except Exception as e:
                print(f"Inference worker error: {e}")
            time.sleep(0.05)

    def _spectral_centroid(self, audio: np.ndarray, sr: int):
        try:
            # magnitude spectrum
            S = np.abs(np.fft.rfft(audio))
            freqs = np.fft.rfftfreq(len(audio), d=1.0/sr)
            if S.sum() == 0:
                return None
            centroid = np.sum(freqs * S) / (S.sum() + 1e-8)
            return float(centroid)
        except Exception:
            return None

    def _spectral_flatness(self, audio: np.ndarray):
        try:
            S = np.abs(np.fft.rfft(audio)) + 1e-12
            geo_mean = np.exp(np.mean(np.log(S)))
            arith_mean = np.mean(S)
            flatness = geo_mean / (arith_mean + 1e-12)
            return float(flatness)
        except Exception:
            return None

    def _harmonic_energy_ratio(self, audio: np.ndarray, sr: int):
        """Estimate harmonic energy ratio (simple peak-to-noise ratio in harmonic bands)."""
        try:
            # Compute short-time FFT and look for harmonic peaks around f0
            S = np.abs(np.fft.rfft(audio))
            if S.sum() == 0:
                return 0.0
            # harmonic energy = energy in top N peaks
            peaks_idx = np.argsort(S)[-10:]
            harmonic_energy = S[peaks_idx].sum()
            total = S.sum()
            return float(harmonic_energy / (total + 1e-8))
        except Exception:
            return None
    
    def set_feedback_options(self, feedback_str):
        """Set which feedback types are enabled (comma-separated string or 'all')."""
        all_types = {"pacing", "volume", "tone", "filler", "pronunciation", "emotion", "visual"}
        if feedback_str.strip().lower() == "all":
            self.enabled_feedback = all_types
        else:
            self.enabled_feedback = set(x.strip().lower() for x in feedback_str.split(",") if x.strip()) & all_types

    def set_language(self, lang_code):
        """Set language code for Whisper transcription."""
        self.language = lang_code
    def _print_ascii_bar(self, values, label, width=40, char='#'):
        """Print a simple ASCII bar graph for a list of values."""
        if not values:
            print(f" {label}: No data")
            return
        import numpy as np
        min_v, max_v = np.min(values), np.max(values)
        rng = max_v - min_v if max_v > min_v else 1
        scaled = [int((v - min_v) / rng * width) for v in values]
        print(f" {label} (min={min_v:.2f}, max={max_v:.2f}):")
        for i, val in enumerate(scaled):
            print(f"  {str(i+1).rjust(3)} | {char * val}")
    
    def _analyze_volume(self):
        """Analyze volume and provide feedback if too quiet."""
        if not self.audio_buffer:
            return
            
        # Calculate RMS of recent audio
        recent_audio = np.concatenate(list(self.audio_buffer)[-10:])  # Last 10 chunks
        rms = np.sqrt(np.mean(recent_audio ** 2))
        self.rms_values.append(rms)
        # Check if volume is too low or too loud
        current_time = time.time()
        if (rms < self.volume_threshold and 
            current_time - self.last_volume_feedback > self.feedback_cooldown):
            self.low_volume_count += 1
            self._provide_feedback("🔊 Project your voice! Your volume is too low.")
            self.last_volume_feedback = current_time
        # Too loud
        if (rms > self.loud_threshold and 
            current_time - self.last_volume_feedback > self.feedback_cooldown):
            self.loud_volume_count += 1
            self._provide_feedback("🔉 You're getting a little loud. Let's keep a conversational tone.")
            self.last_volume_feedback = current_time
    
    def _analyze_pacing(self, current_time: float):
        """Analyze speaking pace and provide feedback if too fast."""
        if len(self.word_timestamps) < 3:
            return
            
        # Calculate WPM from recent words
        recent_words = [item for item in self.word_timestamps 
                       if current_time - item[0] <= 60]  # Last minute
        
        if len(recent_words) >= 2:
            total_words = sum(word_count for _, word_count in recent_words)
            time_span = recent_words[-1][0] - recent_words[0][0]
            if time_span > 0:
                wpm = (total_words / time_span) * 60
                self.max_wpm = max(self.max_wpm, wpm)
                self.min_wpm = min(self.min_wpm, wpm)
                # Check if speaking too fast
                if (wpm > self.wpm_threshold and 
                    current_time - self.last_pacing_feedback > self.feedback_cooldown):
                    self._provide_feedback(f"🐌 Slow down! You're speaking at {wpm:.0f} WPM.")
                    self.last_pacing_feedback = current_time
    
    def _analyze_tone(self):
        """Analyze pitch variation and provide feedback if monotonous."""
        if not self.audio_buffer or len(self.audio_buffer) < 10:
            return
            
        # Simple pitch variation analysis using zero-crossing rate
        recent_audio = np.concatenate(list(self.audio_buffer)[-10:])
        
        # Calculate zero-crossing rate as a proxy for pitch
        zero_crossings = np.where(np.diff(np.signbit(recent_audio)))[0]
        zcr = len(zero_crossings) / len(recent_audio)
        self.pitch_values.append(zcr)
        self.pitch_history.append(zcr)
        if len(self.pitch_values) >= 10:
            # Calculate pitch variation
            pitch_std = np.std(list(self.pitch_values)[-10:])
            current_time = time.time()
            if (pitch_std < self.pitch_variation_threshold and 
                current_time - self.last_tone_feedback > self.feedback_cooldown):
                self.monotone_count += 1
                self._provide_feedback("🎵 Vary your pitch! Your speech sounds monotonous.")
                self.last_tone_feedback = current_time
    def _print_session_review(self):
        """Print a visually improved, detailed summary review of the session, with color, clearer sections, and key stats."""
        import numpy as np
        import sys
        # Color helpers (ANSI)
        def color(text, code):
            if sys.stdout.isatty():
                return f"\033[{code}m{text}\033[0m"
            return text
        BOLD = '1'
        RED = '31;1'
        GREEN = '32;1'
        YELLOW = '33;1'
        BLUE = '34;1'
        CYAN = '36;1'
        MAGENTA = '35;1'
        GREY = '90'

        if not self.session_start_time or not self.session_end_time:
            print(color("No session data to review.", RED))
            return
        duration = self.session_end_time - self.session_start_time
        print("\n" + color("╔══════════════════════════════════════════════╗", CYAN))
        print(color("║           SESSION REVIEW DASHBOARD           ║", CYAN))
        print(color("╚══════════════════════════════════════════════╝", CYAN))
        print(color(f"Duration: {duration:.1f} seconds", BOLD))
        print(color(f"Total Words: {self.total_words}", BOLD))
        if self.max_wpm != float('-inf') and self.min_wpm != float('inf'):
            print(color(f"Max WPM: {self.max_wpm:.1f}", YELLOW))
            print(color(f"Min WPM: {self.min_wpm:.1f}", YELLOW))
        # Volume stats
        if self.rms_values:
            print(color(f"Volume (RMS): avg={np.mean(self.rms_values):.4f} min={np.min(self.rms_values):.4f} max={np.max(self.rms_values):.4f} sd={np.std(self.rms_values):.4f}", BLUE))
            self._print_ascii_bar(self.rms_values[-20:], color("Volume", BLUE), char='=')
        print(color(f"Low Volume Alerts: {self.low_volume_count}", RED if self.low_volume_count else GREEN))
        print(color(f"Loud Volume Alerts: {self.loud_volume_count}", RED if self.loud_volume_count else GREEN))
        # Pitch stats
        if self.pitch_history:
            print(color(f"Pitch (ZCR): avg={np.mean(self.pitch_history):.4f} min={np.min(self.pitch_history):.4f} max={np.max(self.pitch_history):.4f} sd={np.std(self.pitch_history):.4f}", MAGENTA))
            self._print_ascii_bar(self.pitch_history[-20:], color("Pitch", MAGENTA), char='~')
        # WPM bar graph (last 20 WPM samples) and highlight fast/slow segments
        fast_segments = []
        slow_segments = []
        wpm_samples = []
        wpm_threshold = self.wpm_threshold
        slow_threshold = max(80, wpm_threshold * 0.5)  # Customizable lower bound
        if len(self.word_timestamps) > 2:
            for i in range(1, min(21, len(self.word_timestamps))):
                t0, _ = self.word_timestamps[i-1]
                t1, wc = self.word_timestamps[i]
                dt = t1 - t0
                if dt > 0:
                    wpm = (wc / dt) * 60
                    wpm_samples.append(wpm)
                    if wpm > wpm_threshold:
                        fast_segments.append(i)
                    elif wpm < slow_threshold:
                        slow_segments.append(i)
            if wpm_samples:
                # Print WPM bar with fast/slow highlights
                bar = []
                for idx, wpm in enumerate(wpm_samples):
                    if wpm > wpm_threshold:
                        bar.append(color('|', RED))
                    elif wpm < slow_threshold:
                        bar.append(color('|', BLUE))
                    else:
                        bar.append(color('|', YELLOW))
                print(f" {color('WPM', YELLOW)}: " + ''.join(bar) + f"  (Red=fast, Blue=slow, Yellow=ok)")
                # Also print numeric values for context
                print("  WPM values:", ' '.join([f"{int(w)}" for w in wpm_samples]))
                if fast_segments:
                    print(color(f"  ⚠️  Fast segments: {', '.join(str(i+1) for i in fast_segments)} (WPM > {wpm_threshold})", RED))
                if slow_segments:
                    print(color(f"  🐢 Slow segments: {', '.join(str(i+1) for i in slow_segments)} (WPM < {slow_threshold})", BLUE))
                if not fast_segments and not slow_segments:
                    print(color("  ✅ All segments within optimal pace!", GREEN))
        print(color(f"Monotone Alerts: {self.monotone_count}", RED if self.monotone_count else GREEN))
        # Vibe/prosody score (simple: higher pitch/volume SD = more expressive)
        vibe_score = 0
        if self.rms_values and self.pitch_history:
            vibe_score = (np.std(self.rms_values) + np.std(self.pitch_history)) * 50
            print(color(f"Vibe/Prosody Score: {vibe_score:.1f} (higher = more expressive)", CYAN))

        # Actionable advice section
        print(color("\n╔══════════════════════════════════════════════╗", CYAN))
        print(color("║            ACTIONABLE ADVICE                ║", CYAN))
        print(color("╚══════════════════════════════════════════════╝", CYAN))
        # Pacing advice
        if 'wpm_samples' in locals() and wpm_samples:
            fast_count = sum(w > wpm_threshold for w in wpm_samples)
            slow_count = sum(w < slow_threshold for w in wpm_samples)
            if fast_count > 0:
                print(color(f"- Pacing: You spoke too fast in {fast_count} segment(s). Try to pause more and slow down for clarity.", RED))
            if slow_count > 0:
                print(color(f"- Pacing: You spoke too slowly in {slow_count} segment(s). Try to keep a steady, energetic pace.", BLUE))
            if fast_count == 0 and slow_count == 0:
                print(color("- Pacing: Great pacing throughout your speech!", GREEN))
        # Volume advice
        if self.low_volume_count > 0:
            print(color(f"- Volume: Your volume was too low at times. Practice projecting your voice and speaking from your diaphragm.", RED))
        if self.loud_volume_count > 0:
            print(color(f"- Volume: You were too loud at times. Try to moderate your volume for a more pleasant delivery.", YELLOW))
        if self.low_volume_count == 0 and self.loud_volume_count == 0:
            print(color("- Volume: Excellent volume control!", GREEN))
        # Tone advice
        if self.monotone_count > 0:
            print(color(f"- Tone: Your speech was monotonous at times. Add more pitch variation and emotion for engagement.", RED))
        else:
            print(color("- Tone: Good pitch variation and expressiveness!", GREEN))
        # Rhythm advice
        if self.pause_durations:
            std_pause = np.std(self.pause_durations)
            if std_pause > self.irregular_rhythm_threshold:
                print(color("- Rhythm: Your rhythm was irregular. Practice pausing at natural points and keeping a steady flow.", RED))
            else:
                print(color("- Rhythm: Good, even rhythm!", GREEN))
        # Filler word advice
        if self.filler_word_total > 0:
            print(color(f"- Filler Words: Try to reduce filler words like {', '.join(list(self.filler_word_counts.keys())[:3])}. Pause briefly instead of using fillers.", YELLOW))
        else:
            print(color("- Filler Words: No filler words detected. Excellent!", GREEN))
        # Pronunciation advice
        if hasattr(self, "mispronounced_words") and self.mispronounced_words:
            print(color(f"- Pronunciation: Work on clearly pronouncing words like {', '.join(list(set(self.mispronounced_words))[:3])}.", RED))
        else:
            print(color("- Pronunciation: No major issues detected!", GREEN))
        # Emotion/expressiveness
        print(color("\n╔══════════════════════════════════════════════╗", CYAN))
        print(color("║         EMOTION & EXPRESSIVENESS            ║", CYAN))
        print(color("╚══════════════════════════════════════════════╝", CYAN))
        print(color(f"Estimated: {self.emotion_label}", BOLD))
        print(color(f"Expressiveness Score: {self.emotion_score:.3f}", CYAN))
        # Rhythm and pausing analysis
        print(color("\n╔══════════════════════════════════════════════╗", CYAN))
        print(color("║            RHYTHM & PAUSING                 ║", CYAN))
        print(color("╚══════════════════════════════════════════════╝", CYAN))
        if self.pause_durations:
            avg_pause = np.mean(self.pause_durations)
            std_pause = np.std(self.pause_durations)
            print(color(f"Avg Pause: {avg_pause:.2f}s | Stddev: {std_pause:.2f}s", BOLD))
            print(color(f"Long Pauses (> {self.pause_threshold:.1f}s): {self.long_pause_count}", RED if self.long_pause_count else GREEN))
            if std_pause > self.irregular_rhythm_threshold:
                print(color("Rhythm: Irregular (try to keep a more even pace)", RED))
                self.irregular_rhythm_count += 1
            else:
                print(color("Rhythm: Even/regular", GREEN))
        else:
            print(color("Not enough data for rhythm analysis.", GREY))
        # Filler word stats
        print(color("\n╔══════════════════════════════════════════════╗", CYAN))
        print(color("║               FILLER WORDS                  ║", CYAN))
        print(color("╚══════════════════════════════════════════════╝", CYAN))
        if self.filler_word_total > 0:
            print(color(f"Total Filler Words: {self.filler_word_total}", YELLOW))
            for word, count in self.filler_word_counts.items():
                print(color(f" - {word}: {count}", YELLOW))
            print(color("Try to reduce filler words for a more confident delivery!", RED))
        else:
            print(color("No filler words detected. Great job!", GREEN))
        # Pronunciation feedback
        print(color("\n╔══════════════════════════════════════════════╗", CYAN))
        print(color("║         PRONUNCIATION FEEDBACK              ║", CYAN))
        print(color("╚══════════════════════════════════════════════╝", CYAN))
        if hasattr(self, "mispronounced_words") and self.mispronounced_words:
            unique_mispronounced = list(set(self.mispronounced_words))
            print(color(f"Mispronounced/unclear words detected: {len(self.mispronounced_words)}", RED))
            print(color(f" Words: {', '.join(unique_mispronounced[:10])}{'...' if len(unique_mispronounced) > 10 else ''}", RED))
            print(color("Try to pronounce these words more clearly in your next attempt!", RED))
        else:
            print(color("No major pronunciation issues detected.", GREEN))
        # Speech-based grading and transcript
        if self.mode == "speech" and self.reference_text:
            import difflib
            user_text = getattr(self, "transcript_text", " ".join(self.transcript))
            print(color("\nFull Transcript (Whisper):", BOLD))
            # Highlight filler words in transcript
            def highlight_filler(text, filler_words):
                import re
                def repl(match):
                    return color(f"[{match.group(0).upper()}]", YELLOW)
                for word in filler_words:
                    if " " in word:
                        text = text.replace(word, color(f"[{word.upper()}]", YELLOW))
                    else:
                        text = re.sub(rf'\b{re.escape(word)}\b', repl, text, flags=re.IGNORECASE)
                return text
            # Highlight mispronounced words as well
            def highlight_pronunciation(text, mispronounced):
                import re
                for word in set(mispronounced):
                    text = re.sub(rf'\b{re.escape(word)}\b', lambda m: color(f"{{{m.group(0).upper()}}}", RED), text, flags=re.IGNORECASE)
                return text
            highlighted = highlight_filler(user_text, self.filler_words)
            if hasattr(self, "mispronounced_words") and self.mispronounced_words:
                highlighted = highlight_pronunciation(highlighted, self.mispronounced_words)
            print(highlighted)
            print(color("\nReference Speech:", BOLD))
            print(self.reference_text)
            print(color("\nDetailed Comparison:", BOLD))
            # Word-level diff
            ref_words = self.reference_text.split()
            user_words = user_text.split()
            sm = difflib.SequenceMatcher(None, ref_words, user_words)
            opcodes = sm.get_opcodes()
            accuracy_count = 0
            total = 0
            mistakes = []
            for tag, i1, i2, j1, j2 in opcodes:
                if tag == 'equal':
                    accuracy_count += (i2 - i1)
                    total += (i2 - i1)
                elif tag == 'replace':
                    mistakes.append(f"Incorrect: '{' '.join(ref_words[i1:i2])}' → '{' '.join(user_words[j1:j2])}'")
                    total += (i2 - i1)
                elif tag == 'delete':
                    mistakes.append(f"Missing: '{' '.join(ref_words[i1:i2])}'")
                    total += (i2 - i1)
                elif tag == 'insert':
                    mistakes.append(f"Extra: '{' '.join(user_words[j1:j2])}'")
            accuracy = accuracy_count / max(1, total)
            print(color(f"\nAccuracy: {accuracy*100:.1f}%", BOLD))
            if mistakes:
                print(color("\nMistakes:", RED))
                for m in mistakes:
                    print(color(f"- {m}", RED))
            else:
                print(color("No mistakes detected!", GREEN))
            # Text summary
            print(color("\nSummary:", BOLD))
            if accuracy > 0.95:
                print(color("Excellent! Your recitation was very accurate. Keep practicing for even more fluency.", GREEN))
            elif accuracy > 0.8:
                print(color("Good job! Review the mistakes above and try to reduce them in your next attempt.", YELLOW))
            else:
                print(color("Needs improvement. Focus on reading carefully and matching the reference speech word for word.", RED))
    
    def _provide_feedback(self, message: str):
        """Provide audio and text feedback to the user."""
        if self.mode == "speech":
            # No real-time feedback in speech mode
            return
        print(f" FEEDBACK: {message}")
        # Use TTS in a separate thread to avoid blocking
        if self.tts_available:
            def speak():
                try:
                    self.tts_engine.say(message.split("! ")[-1])  # Remove emoji and speak the main message
                    self.tts_engine.runAndWait()
                except Exception as e:
                    print(f"  TTS error: {e}")
            tts_thread = threading.Thread(target=speak)
            tts_thread.daemon = True
            tts_thread.start()
        # Speech-based grading
        if self.mode == "speech" and self.reference_text:
            print("\n Speech Comparison:")
            user_text = " ".join(self.transcript)
            import difflib
            sm = difflib.SequenceMatcher(None, self.reference_text.split(), user_text.split())
            match = sm.ratio()
            print(f" Speech Match: {match*100:.1f}%")
            # Show missing/extra words (optional, simple diff)
            ref_words = set(self.reference_text.split())
            user_words = set(user_text.split())
            missing = ref_words - user_words
            extra = user_words - ref_words
            print(f" Missing words: {', '.join(list(missing)[:10])}{'...' if len(missing)>10 else ''}")
            print(f" Extra words: {', '.join(list(extra)[:10])}{'...' if len(extra)>10 else ''}")
            # Suggest corrections
            if match < 0.9:
                print(" Suggestion: Practice reading the speech aloud, focusing on accuracy and pacing.")
            else:
                print(" Great job! Your recitation closely matches the reference.")
    
    def _print_live_metrics(self):
        """Print live metrics for debugging."""
        # Calculate current WPM
        current_time = time.time()
        recent_words = [item for item in self.word_timestamps 
                       if current_time - item[0] <= 30]  # Last 30 seconds
        
        wpm = 0
        if len(recent_words) >= 2:
            total_words = sum(word_count for _, word_count in recent_words)
            time_span = recent_words[-1][0] - recent_words[0][0]
            if time_span > 0:
                wpm = (total_words / time_span) * 60
        
        # Initialize metrics
        rms = 0
        pitch_variation = 0
        
        # Calculate volume and pitch metrics if available
        if PYAUDIO_AVAILABLE and self.audio_buffer:
            recent_audio = np.concatenate(list(self.audio_buffer)[-5:])  # Last 5 chunks
            rms = np.sqrt(np.mean(recent_audio ** 2))
            
            if len(self.pitch_values) >= 5:
                pitch_variation = np.std(list(self.pitch_values)[-5:])
        
        # Print metrics
        if PYAUDIO_AVAILABLE:
            print(f" Volume: {rms:.4f} | WPM: {wpm:.0f} | Pitch Var: {pitch_variation:.4f}")
        else:
            print(f" WPM: {wpm:.0f} | Volume: N/A | Pitch Var: N/A")


def main():
    """Main entry point for the speech coach."""
    coach = SpeechCoach()
    
    try:
        coach.start()
    except KeyboardInterrupt:
        print("\n User interrupted")
    except Exception as e:
        print(f" Error: {e}")
    finally:
        coach.stop()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
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


import threading
import sys
import time
import select

import tempfile
import os


class SpeechCoach:
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
        self.loud_threshold = 0.2  # RMS threshold for too loud (customizable)

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
        
    def start(self):
        """Start the speech coaching session with keyboard triggers."""
        print("🎤 Speech Coach Ready!")
        if not self.microphone_available:
            print(" Cannot start: No microphone available")
            return
        print("Calibrating microphone...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        print(" Calibration complete!")
        print("Controls: [r] Start  [p] Pause/Resume  [s] Stop  [Ctrl+C] Quit")
        print("=" * 50)
        self.running = True
        self.paused = True
        self.session_active = False
        self.key_thread = threading.Thread(target=self._key_listener)
        self.key_thread.daemon = True
        self.key_thread.start()
        if self.mode == "speech":
            # Prepare temp WAV file for recording
            self.audio_record_path = tempfile.mktemp(suffix=".wav")
            self.audio_thread = threading.Thread(target=self._record_audio_wav)
            self.audio_thread.daemon = True
            self.audio_thread.start()
        elif PYAUDIO_AVAILABLE:
            self.audio_thread = threading.Thread(target=self._audio_capture_loop)
            self.audio_thread.daemon = True
            self.audio_thread.start()
        self.analysis_thread = threading.Thread(target=self._analysis_loop)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        self._speech_recognition_loop()

    def _record_audio_wav(self):
        """Record the session to a WAV file for Whisper transcription (speech mode only)."""
        import soundfile as sf
        import pyaudio
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        frames = []
        while self.running:
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
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
        model = whisper.load_model("base")
        result = model.transcribe(self.audio_record_path, word_timestamps=True, verbose=False)
        # result["text"] is always a string
        self.transcript_text = str(result["text"]) if result["text"] is not None else ""
        self.transcript = self.transcript_text.split() if self.transcript_text else []
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
            while self.running:
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.float32)
                    self.audio_buffer.append(audio_data)
                except Exception as e:
                    print(f" Audio capture error: {e}")
                    break
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            p.terminate()
    
    def _speech_recognition_loop(self):
        """Continuous speech recognition for WPM calculation."""
        while self.running:
            if not self.session_active or self.paused:
                time.sleep(0.1)
                continue
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                try:
                    text = self.recognizer.recognize_google(audio)
                    if text.strip():
                        word_count = len(text.split())
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
                            self.transcript.append(text)
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    print(f" Speech recognition error: {e}")
            except sr.WaitTimeoutError:
                pass
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f" Recognition loop error: {e}")
                time.sleep(0.1)
    
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
                # Analyze emotion/expressiveness (pitch/volume stats)
                self._analyze_emotion()
                # Print live metrics
                self._print_live_metrics()
                time.sleep(1)
            except Exception as e:
                print(f" Analysis error: {e}")
                time.sleep(1)

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
        """Print a detailed summary review of the session, including transcript and speech comparison in speech mode."""
        import numpy as np
        if not self.session_start_time or not self.session_end_time:
            print("No session data to review.")
            return
        duration = self.session_end_time - self.session_start_time
        print("\n SESSION REVIEW")
        print("=" * 40)
        print(f" Duration: {duration:.1f} seconds")
        print(f" Total Words: {self.total_words}")
        if self.max_wpm != float('-inf') and self.min_wpm != float('inf'):
            print(f" Max WPM: {self.max_wpm:.1f}")
            print(f" Min WPM: {self.min_wpm:.1f}")
        # Volume stats
        if self.rms_values:
            print(f" Volume (RMS): avg={np.mean(self.rms_values):.4f} min={np.min(self.rms_values):.4f} max={np.max(self.rms_values):.4f} sd={np.std(self.rms_values):.4f}")
            self._print_ascii_bar(self.rms_values[-20:], "Volume", char='=')
        print(f" Low Volume Alerts: {self.low_volume_count}")
        print(f" Loud Volume Alerts: {self.loud_volume_count}")
        # Pitch stats
        if self.pitch_history:
            print(f" Pitch (ZCR): avg={np.mean(self.pitch_history):.4f} min={np.min(self.pitch_history):.4f} max={np.max(self.pitch_history):.4f} sd={np.std(self.pitch_history):.4f}")
            self._print_ascii_bar(self.pitch_history[-20:], "Pitch", char='~')
        # WPM bar graph (last 20 WPM samples)
        if len(self.word_timestamps) > 2:
            wpm_samples = []
            for i in range(1, min(21, len(self.word_timestamps))):
                t0, _ = self.word_timestamps[i-1]
                t1, wc = self.word_timestamps[i]
                dt = t1 - t0
                if dt > 0:
                    wpm = (wc / dt) * 60
                    wpm_samples.append(wpm)
            if wpm_samples:
                self._print_ascii_bar(wpm_samples, "WPM", char='|')
        print(f" Monotone Alerts: {self.monotone_count}")
        # Vibe/prosody score (simple: higher pitch/volume SD = more expressive)
        vibe_score = 0
        if self.rms_values and self.pitch_history:
            vibe_score = (np.std(self.rms_values) + np.std(self.pitch_history)) * 50
            print(f" Vibe/Prosody Score: {vibe_score:.1f} (higher = more expressive)")
        # Emotion/expressiveness
        print("=" * 40)
        print(" Emotion & Expressiveness:")
        print(f"  Estimated: {self.emotion_label}")
        print(f"  Expressiveness Score: {self.emotion_score:.3f}")
        # Rhythm and pausing analysis
        print("=" * 40)
        print(" Rhythm & Pausing:")
        if self.pause_durations:
            avg_pause = np.mean(self.pause_durations)
            std_pause = np.std(self.pause_durations)
            print(f"  Avg Pause: {avg_pause:.2f}s | Stddev: {std_pause:.2f}s")
            print(f"  Long Pauses (> {self.pause_threshold:.1f}s): {self.long_pause_count}")
            if std_pause > self.irregular_rhythm_threshold:
                print("  Rhythm: Irregular (try to keep a more even pace)")
                self.irregular_rhythm_count += 1
            else:
                print("  Rhythm: Even/regular")
        else:
            print("  Not enough data for rhythm analysis.")
        # Filler word stats
        print("=" * 40)
        print(" Filler Words:")
        if self.filler_word_total > 0:
            print(f"  Total Filler Words: {self.filler_word_total}")
            for word, count in self.filler_word_counts.items():
                print(f"   - {word}: {count}")
            print("  Try to reduce filler words for a more confident delivery!")
        else:
            print("  No filler words detected. Great job!")
        # Pronunciation feedback
        print("=" * 40)
        print(" Pronunciation Feedback:")
        if hasattr(self, "mispronounced_words") and self.mispronounced_words:
            unique_mispronounced = list(set(self.mispronounced_words))
            print(f"  Mispronounced/unclear words detected: {len(self.mispronounced_words)}")
            print(f"   Words: {', '.join(unique_mispronounced[:10])}{'...' if len(unique_mispronounced) > 10 else ''}")
            print("  Try to pronounce these words more clearly in your next attempt!")
        else:
            print("  No major pronunciation issues detected.")
        print("=" * 40)

        # Speech-based grading and transcript
        if self.mode == "speech" and self.reference_text:
            import difflib
            user_text = getattr(self, "transcript_text", " ".join(self.transcript))
            print("\n Full Transcript (Whisper):")
            # Highlight filler words in transcript
            def highlight_filler(text, filler_words):
                import re
                def repl(match):
                    return f"[{match.group(0).upper()}]"
                for word in filler_words:
                    if " " in word:
                        text = text.replace(word, f"[{word.upper()}]")
                    else:
                        text = re.sub(rf'\b{re.escape(word)}\b', repl, text, flags=re.IGNORECASE)
                return text
            # Highlight mispronounced words as well
            def highlight_pronunciation(text, mispronounced):
                import re
                for word in set(mispronounced):
                    text = re.sub(rf'\b{re.escape(word)}\b', lambda m: f"{{{m.group(0).upper()}}}", text, flags=re.IGNORECASE)
                return text
            highlighted = highlight_filler(user_text, self.filler_words)
            if hasattr(self, "mispronounced_words") and self.mispronounced_words:
                highlighted = highlight_pronunciation(highlighted, self.mispronounced_words)
            print(highlighted)
            print("\n Reference Speech:")
            print(self.reference_text)
            print("\n Detailed Comparison:")
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
            print(f"\n Accuracy: {accuracy*100:.1f}%")
            if mistakes:
                print("\n Mistakes:")
                for m in mistakes: # Can be long, so limit output?
                    print(f"- {m}")
                # if len(mistakes) > 10:
                #     print(f"...and {len(mistakes)-10} more.")
            else:
                print(" No mistakes detected!")
            # Text summary
            print("\n Summary:")
            if accuracy > 0.95:
                print("Excellent! Your recitation was very accurate. Keep practicing for even more fluency.")
            elif accuracy > 0.8:
                print("Good job! Review the mistakes above and try to reduce them in your next attempt.")
            else:
                print("Needs improvement. Focus on reading carefully and matching the reference speech word for word.")
    
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
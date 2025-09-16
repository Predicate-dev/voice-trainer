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

class SpeechCoach:
    """Real-time speech analysis and coaching system with start/pause/stop triggers and session review."""
    
    def __init__(self):
        # Speech recognition setup
        self.recognizer = sr.Recognizer()
        try:
            self.microphone = sr.Microphone()
            self.microphone_available = True
        except Exception as e:
            print(f"⚠️  Microphone not available: {e}")
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
        self.monotone_count = 0

        # Feedback cooldowns (prevent spam)
        self.last_pacing_feedback = 0
        self.last_volume_feedback = 0
        self.last_tone_feedback = 0
        self.feedback_cooldown = 5  # seconds
        
    def start(self):
        """Start the speech coaching session with keyboard triggers."""
        print("🎤 Speech Coach Ready!")
        if not self.microphone_available:
            print("❌ Cannot start: No microphone available")
            return
        print("Calibrating microphone...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        print("✅ Calibration complete!")
        print("Controls: [r] Start  [p] Pause/Resume  [s] Stop  [Ctrl+C] Quit")
        print("=" * 50)
        self.running = True
        self.paused = True
        self.session_active = False
        self.key_thread = threading.Thread(target=self._key_listener)
        self.key_thread.daemon = True
        self.key_thread.start()
        if PYAUDIO_AVAILABLE:
            self.audio_thread = threading.Thread(target=self._audio_capture_loop)
            self.audio_thread.daemon = True
            self.audio_thread.start()
        self.analysis_thread = threading.Thread(target=self._analysis_loop)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        self._speech_recognition_loop()
    
    def stop(self):
        """Stop the speech coaching session and print review."""
        print("\n🛑 Stopping Speech Coach...")
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
        self._print_session_review()

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
                print("▶️  Session started!")
                self.session_active = True
                self.paused = False
                self.session_start_time = time.time()
            elif self.paused:
                print("▶️  Resumed.")
                self.paused = False
        elif key == 'p':
            if self.session_active and not self.paused:
                print("⏸️  Paused.")
                self.paused = True
        elif key == 's':
            print("🛑 Stop key pressed.")
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
                    print(f"⚠️  Audio capture error: {e}")
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
                        print(f"📝 Recognized: {text}")
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    print(f"⚠️  Speech recognition error: {e}")
            except sr.WaitTimeoutError:
                pass
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"⚠️  Recognition loop error: {e}")
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
                # Print live metrics
                self._print_live_metrics()
                time.sleep(1)
            except Exception as e:
                print(f"⚠️  Analysis error: {e}")
                time.sleep(1)
    
    def _analyze_volume(self):
        """Analyze volume and provide feedback if too quiet."""
        if not self.audio_buffer:
            return
            
        # Calculate RMS of recent audio
        recent_audio = np.concatenate(list(self.audio_buffer)[-10:])  # Last 10 chunks
        rms = np.sqrt(np.mean(recent_audio ** 2))
        
        # Check if volume is too low
        current_time = time.time()
        if (rms < self.volume_threshold and 
            current_time - self.last_volume_feedback > self.feedback_cooldown):
            self.low_volume_count += 1
            self._provide_feedback("🔊 Project your voice! Your volume is too low.")
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
        """Print a summary review of the session."""
        if not self.session_start_time or not self.session_end_time:
            print("No session data to review.")
            return
        duration = self.session_end_time - self.session_start_time
        print("\n📋 SESSION REVIEW")
        print("=" * 40)
        print(f"🕒 Duration: {duration:.1f} seconds")
        print(f"📝 Total Words: {self.total_words}")
        if self.max_wpm != float('-inf') and self.min_wpm != float('inf'):
            print(f"📈 Max WPM: {self.max_wpm:.1f}")
            print(f"📉 Min WPM: {self.min_wpm:.1f}")
        print(f"🔊 Low Volume Alerts: {self.low_volume_count}")
        print(f"🎵 Monotone Alerts: {self.monotone_count}")
        print("=" * 40)
    
    def _provide_feedback(self, message: str):
        """Provide audio and text feedback to the user."""
        print(f"💬 FEEDBACK: {message}")
        
        # Use TTS in a separate thread to avoid blocking
        if self.tts_available:
            def speak():
                try:
                    self.tts_engine.say(message.split("! ")[-1])  # Remove emoji and speak the main message
                    self.tts_engine.runAndWait()
                except Exception as e:
                    print(f"⚠️  TTS error: {e}")
            
            tts_thread = threading.Thread(target=speak)
            tts_thread.daemon = True
            tts_thread.start()
    
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
            print(f"📊 Volume: {rms:.4f} | WPM: {wpm:.0f} | Pitch Var: {pitch_variation:.4f}")
        else:
            print(f"📊 WPM: {wpm:.0f} | Volume: N/A | Pitch Var: N/A")


def main():
    """Main entry point for the speech coach."""
    coach = SpeechCoach()
    
    try:
        coach.start()
    except KeyboardInterrupt:
        print("\n🛑 User interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        coach.stop()


if __name__ == "__main__":
    main()
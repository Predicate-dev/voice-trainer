import sys
import os
import threading
import time
import tempfile
import io
import contextlib
import re

import soundfile
import numpy as np
import pyttsx3

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit, QHBoxLayout, QProgressBar, QFileDialog, QComboBox, QCheckBox, QSlider
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtMultimedia import QSound

from speech_coach import SpeechCoach

class SpeechCoachApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech Coach Desktop")
        self.setGeometry(200, 200, 600, 500)
        self.setFont(QFont('Arial', 11))
        self.coach = None
        self.session_thread = None
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_status)
        self.session_active = False

    def init_ui(self):
        layout = QVBoxLayout()
        self.status_label = QLabel("Welcome to Speech Coach! Press Start to begin.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    # Options: Mode, Reference Speech, Recording, and direct input
        options_layout = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["freestyle", "speech"])
        options_layout.addWidget(QLabel("Mode:"))
        options_layout.addWidget(self.mode_combo)
        self.ref_btn = QPushButton("Select Speech File")
        self.ref_btn.clicked.connect(self.select_reference)
        self.ref_label = QLabel("")
        options_layout.addWidget(self.ref_btn)
        options_layout.addWidget(self.ref_label)
        layout.addLayout(options_layout)

        # Direct input for custom speech
        self.speech_input = QTextEdit()
        self.speech_input.setPlaceholderText("Or paste/type your speech here (used in 'speech' mode)")
        self.speech_input.setFixedHeight(60)
        layout.addWidget(self.speech_input)

        # Recording option
        self.record_checkbox = QCheckBox("Enable Recording (speech mode)")
        self.record_checkbox.setChecked(True)
        layout.addWidget(self.record_checkbox)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Session")
        self.start_btn.clicked.connect(self.start_session)
        btn_layout.addWidget(self.start_btn)
        self.stop_btn = QPushButton("Stop Session")
        self.stop_btn.clicked.connect(self.stop_session)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.live_metrics = QLabel("")
        layout.addWidget(self.live_metrics)

        self.review_box = QTextEdit()
        self.review_box.setReadOnly(True)
        layout.addWidget(self.review_box)

        self.setLayout(layout)

        self.reference_text = None

        tts_layout = QHBoxLayout()
        tts_layout.addWidget(QLabel("TTS Speed:"))
        self.tts_speed = QSlider(Qt.Horizontal)
        self.tts_speed.setMinimum(100)
        self.tts_speed.setMaximum(300)
        self.tts_speed.setValue(200)
        self.tts_speed.setTickInterval(10)
        self.tts_speed.setTickPosition(QSlider.TicksBelow)
        tts_layout.addWidget(self.tts_speed)
        tts_layout.addWidget(QLabel("TTS Volume:"))
        self.tts_volume = QSlider(Qt.Horizontal)
        self.tts_volume.setMinimum(0)
        self.tts_volume.setMaximum(100)
        self.tts_volume.setValue(80)
        self.tts_volume.setTickInterval(10)
        self.tts_volume.setTickPosition(QSlider.TicksBelow)
        tts_layout.addWidget(self.tts_volume)
        tts_layout.addWidget(QLabel("TTS Pitch:"))
        self.tts_pitch = QSlider(Qt.Horizontal)
        self.tts_pitch.setMinimum(50)
        self.tts_pitch.setMaximum(200)
        self.tts_pitch.setValue(100)
        self.tts_pitch.setTickInterval(10)
        self.tts_pitch.setTickPosition(QSlider.TicksBelow)
        tts_layout.addWidget(self.tts_pitch)
        self.preview_btn = QPushButton("Preview Model Speech")
        self.preview_btn.clicked.connect(self.preview_model_speech)
        tts_layout.addWidget(self.preview_btn)
        layout.addLayout(tts_layout)

    def select_reference(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Reference Speech", "", "Text Files (*.txt);;All Files (*)")
        if fname:
            try:
                with open(fname, "r") as f:
                    self.reference_text = f.read()
                self.ref_label.setText(f"Loaded: {fname.split('/')[-1]}")
            except Exception as e:
                self.ref_label.setText(f"Error loading file: {e}")

    def start_session(self):
        mode = self.mode_combo.currentText()
        # Priority: direct input > file
        custom_speech = self.speech_input.toPlainText().strip()
        reference = None
        if mode == "speech":
            if custom_speech:
                reference = custom_speech
            elif self.reference_text:
                reference = self.reference_text
        enable_recording = self.record_checkbox.isChecked()
        self.status_label.setText(f"Session started. Mode: {mode}. Speak into your microphone.")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.review_box.clear()
        self.progress.setValue(0)
        self.session_active = True
        # Pass mode, reference, and recording option to SpeechCoach
        self.coach = SpeechCoach(mode=mode, reference_text=reference)
        self._live_transcript = []
        # Always create the thread before setting daemon/start
        self.session_thread = threading.Thread(target=self.run_coach_gui)
        if self.session_thread is not None:
            self.session_thread.daemon = True
            self.session_thread.start()
        self.timer.start(1000)
    
    def run_coach_gui(self) -> None:
        try:
            if self.coach is not None:
                self.coach.start(gui_mode=True)
            else:
                self.status_label.setText("Error: SpeechCoach is not initialized.")
        except Exception as e:
            self.status_label.setText(f"Error: {e}")

    def preview_model_speech(self):
        """Synthesize and play the input speech with selected TTS parameters, saving to WAV and playing with sounddevice."""
        import pyttsx3
        import tempfile
        import os
        import platform
        import soundfile as sf
        import sounddevice as sd
        text = self.speech_input.toPlainText().strip()
        if not text:
            self.status_label.setText("Please enter or paste a speech to preview.")
            return
        engine = pyttsx3.init()
        engine.setProperty('rate', self.tts_speed.value())
        engine.setProperty('volume', self.tts_volume.value() / 100.0)
        # Select a male voice if available
        voices = engine.getProperty('voices')
        male_voice_id = None
        # Try to iterate voices, else print debug info
        # Use the user-specified voice
        engine.setProperty('voice', 'com.apple.speech.synthesis.voice.Cellos')

        print("[DEBUG] Listing available voices:")
        try:
            voices = engine.getProperty('voices')
            if hasattr(voices, '__iter__') and not isinstance(voices, str):
                for v in voices:
                    print(f"Voice ID: {getattr(v, 'id', v)} | Name: {getattr(v, 'name', '?')} | Gender: {getattr(v, 'gender', '?')} | Lang: {getattr(v, 'languages', '?')}")
            else:
                print(f"[DEBUG] voices is not iterable: type={type(voices)}, value={voices}")
        except Exception as e:
            print(f"[DEBUG] Error listing voices: {e}")
        # Only set pitch if not on macOS (NSSS does not support pitch)
        if platform.system() != 'Darwin':
            try:
                engine.setProperty('pitch', self.tts_pitch.value())
            except Exception:
                pass
        # Save to temp WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tf:
            wav_path = tf.name
        engine.save_to_file(text, wav_path)
        engine.runAndWait()
        self.status_label.setText(f"Playing model speech at speed {self.tts_speed.value()}, volume {self.tts_volume.value()}, pitch {self.tts_pitch.value()}.")
        # Play using sounddevice
        try:
            # Print available sound devices for debugging
            devices = sd.query_devices()
            print("Available sound devices:")
            for idx, dev in enumerate(devices):
                # Use attribute access (sounddevice returns namedtuple or similar)
                name = getattr(dev, 'name', str(dev))
                max_out = getattr(dev, 'max_output_channels', '?')
                print(f"{idx}: {name} (output channels: {max_out})")
            data, samplerate = sf.read(wav_path, dtype='float32')
            if data is None or data.size == 0:
                self.status_label.setText("TTS generated an empty or invalid audio file.")
                print("[DEBUG] WAV file is empty or invalid.")
            else:
                print(f"[DEBUG] Playing audio: shape={data.shape}, samplerate={samplerate}")
                sd.play(data, samplerate)
                sd.wait()
                self.status_label.setText("Model speech playback finished.")
        except Exception as e:
            self.status_label.setText(f"Audio playback error: {e}")
            print(f"[DEBUG] Audio playback error: {e}")
        try:
            os.remove(wav_path)
        except Exception:
            pass

    def run_coach(self):
        try:
            if self.coach is not None:
                self.coach.start()
            else:
                self.status_label.setText("Error: SpeechCoach is not initialized.")
        except Exception as e:
            self.status_label.setText(f"Error: {e}")

    def stop_session(self):
        if self.coach:
            self.coach.stop()
        self.status_label.setText("Session stopped. Review below.")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.session_active = False
        self.timer.stop()
        # Show review
        self.show_review()

    def update_status(self):
        if self.coach and self.session_active:
            # Show live metrics if available
            try:
                current_time = time.time()
                recent_words = [item for item in self.coach.word_timestamps if current_time - item[0] <= 30]
                wpm = 0
                if len(recent_words) >= 2:
                    total_words = sum(word_count for _, word_count in recent_words)
                    time_span = recent_words[-1][0] - recent_words[0][0]
                    if time_span > 0:
                        wpm = (total_words / time_span) * 60
                rms = 0
                if hasattr(self.coach, 'audio_buffer') and self.coach.audio_buffer:
                    import numpy as np
                    recent_audio = np.concatenate(list(self.coach.audio_buffer)[-5:])
                    rms = np.sqrt(np.mean(recent_audio ** 2))
                self.live_metrics.setText(f"Live WPM: {wpm:.0f} | Volume: {rms:.3f}")
                self.progress.setValue(min(100, int(self.coach.total_words)))
                # Show live transcript (for both modes)
                transcript = None
                if hasattr(self.coach, 'transcript') and self.coach.transcript:
                    transcript = self.coach.transcript
                elif hasattr(self.coach, 'word_timestamps') and self.coach.word_timestamps:
                    # fallback: not as accurate
                    transcript = [w for t, w in self.coach.word_timestamps]
                if transcript:
                    # Only show last 30 words for brevity
                    live_text = ' '.join(transcript[-30:])
                    self.review_box.setPlainText(f"[Live Transcript]\n{live_text}\n\n" + self.review_box.toPlainText().split('[Live Transcript]')[-1])
            except Exception:
                pass

    def show_review(self):
        if self.coach:
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.coach._print_session_review()
            text = buf.getvalue()
            # Simple color scheme: blue for headers, green for good, red for alerts, yellow for warnings
            import re
            html = text
            # Section headers
            html = re.sub(r'(╔[═]+╗)', r'<br><span style="color:#1976d2;font-weight:bold;">[0m\1</span>', html)
            html = re.sub(r'(║[\sA-Z&]+║)', r'<span style="color:#1976d2;font-weight:bold;">[0m\1</span>', html)
            html = re.sub(r'(╚[═]+╝)', r'<span style="color:#1976d2;font-weight:bold;">[0m\1</span><br>', html)
            # Good/positive
            html = re.sub(r'(Excellent|Great job|No filler words detected|No major issues detected|Good pitch variation|Good, even rhythm|Excellent volume control|No mistakes detected)', r'<span style="color:#388e3c;font-weight:bold;">\1</span>', html, flags=re.IGNORECASE)
            # Alerts/negative
            html = re.sub(r'(Too low|Too loud|monotonous|Irregular|Try to|Missing|Incorrect|Needs improvement|Fast segments|Slow segments|Pronounce|Mistakes:)', r'<span style="color:#d32f2f;font-weight:bold;">\1</span>', html, flags=re.IGNORECASE)
            # Warnings/neutral
            html = re.sub(r'(Good job|Review the mistakes|Summary:|Accuracy:)', r'<span style="color:#fbc02d;font-weight:bold;">\1</span>', html, flags=re.IGNORECASE)
            # Remove ANSI codes if present
            html = re.sub(r'\u001b\[[0-9;]*m', '', html)
            # Replace newlines with <br>
            html = html.replace('\n', '<br>')
            self.review_box.setHtml(f'<pre style="font-family:monospace;">{html}</pre>')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpeechCoachApp()
    window.show()
    sys.exit(app.exec_())

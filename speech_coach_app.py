import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit, QHBoxLayout, QProgressBar, QFileDialog
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
import threading
import time

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
        from PyQt5.QtWidgets import QComboBox, QCheckBox, QTextEdit
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
        # Always record in speech mode for GUI (let user ignore file if not needed)
        self.session_thread = threading.Thread(target=self.run_coach_gui)
        self.session_thread.daemon = True
        self.session_thread.start()
        self.timer.start(1000)

    def run_coach_gui(self):
        try:
            self.coach.start(gui_mode=True)
        except Exception as e:
            self.status_label.setText(f"Error: {e}")

    def run_coach(self):
        try:
            self.coach.start()
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
            self.review_box.setPlainText(buf.getvalue())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpeechCoachApp()
    window.show()
    sys.exit(app.exec_())

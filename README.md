# Voice Trainer - Speech Coach PoC

A Python command-line speech coaching system that provides real-time feedback on speaking skills. This proof-of-concept uses speech recognition, audio analysis, and text-to-speech to help users improve their speaking pace, volume, and tone variation.

## Features

- **Pacing Analysis**: Measures words per minute (WPM) and provides feedback if speaking too fast
- **Volume Analysis**: Calculates RMS volume and provides feedback if speaking too quietly
- **Tone Analysis**: Analyzes pitch variation and provides feedback if speech is monotonous
- **Real-time Feedback**: Provides both audio and visual feedback during speech
- **Live Metrics**: Displays debugging metrics for volume, WPM, and pitch variation
- **Keyboard Controls**: Start, pause/resume, and stop your session with keys ([r] Start, [p] Pause/Resume, [s] Stop)
- **Session Review**: Get a summary of your session (duration, words, alerts) at the end

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Predicate-dev/voice-trainer.git
cd voice-trainer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: On some systems, you may need to install additional system dependencies for PyAudio:
- **Ubuntu/Debian**: `sudo apt-get install portaudio19-dev python3-pyaudio`
- **macOS**: `brew install portaudio`
- **Windows**: PyAudio should install directly via pip

**Optional**: For text-to-speech feedback, install system TTS:
- **Ubuntu/Debian**: `sudo apt-get install espeak espeak-data`
- **macOS**: Built-in TTS should work
- **Windows**: Built-in TTS should work

## Project Structure

```
voice-trainer/
├── speech_coach.py     # Core speech analysis and coaching engine
├── main.py            # CLI entry point with argument parsing
├── demo.py            # Feature demonstration script
├── test_speech_coach.py # Test suite with simulation
├── requirements.txt   # Python dependencies
├── config.ini         # Configuration file for thresholds
├── README.md          # Documentation
└── .gitignore         # Git ignore file
```

## Quick Start

1. **Run the demo** to see what the system can do:
   ```bash
   python demo.py
   ```

2. **Test the core functionality** without microphone:
   ```bash
   python test_speech_coach.py
   ```

3. **Start the speech coach** (requires microphone):
   ```bash
   python main.py
   ```
   
   **Controls during session:**
   - [r] Start
   - [p] Pause/Resume
   - [s] Stop (shows session review)
   - [Ctrl+C] Quit

### Basic Usage
```bash
python main.py
```

### With Custom Thresholds
```bash
python main.py --wpm-threshold 150 --volume-threshold 0.02 --pitch-threshold 0.3
```

### Direct Module Usage
```bash
python speech_coach.py
```

## How It Works

1. **Microphone Calibration**: The system calibrates to your microphone and ambient noise
2. **Real-time Analysis**: Continuously analyzes your speech for:
   - Speaking pace (words per minute)
   - Volume levels (RMS calculation)
   - Pitch variation (tone analysis)
3. **Session Controls**: Use keyboard keys to start, pause, resume, or stop your session at any time.
4. **Feedback**: Provides immediate feedback when thresholds are exceeded:
   - "Slow down!" if speaking too fast (default: >180 WPM)
   - "Project your voice!" if speaking too quietly
   - "Vary your pitch!" if speech is monotonous
5. **Live Metrics**: Displays current metrics for debugging and monitoring
6. **Session Review**: When you stop, a summary is shown: duration, total words, max/min WPM, and alert counts.

## Modes

- **Freestyle Mode**: Speak freely and get real-time feedback on pacing, volume, and tone.
- **Speech Mode**: Provide a reference speech (text file), recite it, and get a detailed accuracy report and feedback at the end. Uses OpenAI Whisper for highly accurate transcription.

### Speech Mode Example
```bash
python main.py --mode speech --reference-speech speech.txt
```
- At the end, you'll see:
  - The full transcript of your speech (via Whisper)
  - A word-level comparison with the reference
  - Highlighted mistakes (missing, extra, or incorrect words)
  - An overall accuracy score and summary

## Configuration

You can adjust the feedback thresholds:

- `--wpm-threshold`: Words per minute threshold (default: 180)
- `--volume-threshold`: RMS volume threshold (default: 0.01)
- `--pitch-threshold`: Pitch variation threshold (default: 0.2)

## Requirements

- Python 3.7+
- Microphone access
- Internet connection (for speech recognition)

## Dependencies

- `speech_recognition`: For speech-to-text conversion
- `pyaudio`: For real-time audio capture
- `pyttsx3`: For text-to-speech feedback
- `numpy`: For audio signal processing
- `openai-whisper`: For accurate offline transcription in speech mode
- `soundfile`: For saving audio for Whisper

## Troubleshooting

- **Microphone not detected**: Ensure your microphone is connected and accessible
- **PyAudio installation issues**: Install system dependencies as mentioned in installation
- **Speech recognition errors**: Check internet connection and microphone quality
- **No feedback**: Adjust thresholds or check microphone sensitivity

## Future Enhancements

- Advanced pitch analysis using FFT
- Filler word detection ("um", "uh", etc.)
- Speaking rhythm analysis
- Web-based interface
- Training session recording and playback

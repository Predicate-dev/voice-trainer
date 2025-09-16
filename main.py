#!/usr/bin/env python3
"""
Voice Trainer CLI - Command-line interface for the speech coach.
"""

import argparse
import sys
from speech_coach import SpeechCoach


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Voice Trainer - Real-time speech coaching system",
        epilog="Press Ctrl+C to stop the session at any time."
    )

    parser.add_argument(
        "--mode",
        choices=["freestyle", "speech"],
        default="freestyle",
        help="Mode: 'freestyle' for live feedback, 'speech' for speech-based grading."
    )

    parser.add_argument(
        "--reference-speech",
        type=str,
        help="Path to text file containing reference speech (required for speech mode)"
    )

    parser.add_argument(
        "--wpm-threshold",
        type=int,
        default=180,
        help="Words per minute threshold for pacing feedback (default: 180)"
    )

    parser.add_argument(
        "--volume-threshold",
        type=float,
        default=0.01,
        help="RMS volume threshold for volume feedback (default: 0.01)"
    )

    parser.add_argument(
        "--pitch-threshold",
        type=float,
        default=0.2,
        help="Pitch variation threshold for tone feedback (default: 0.2)"
    )

    args = parser.parse_args()
    
    print("🎤 Voice Trainer - Speech Coach PoC")
    print("=" * 40)
    print(f"📈 WPM Threshold: {args.wpm_threshold}")
    print(f"🔊 Volume Threshold: {args.volume_threshold}")
    print(f"🎵 Pitch Threshold: {args.pitch_threshold}")
    print(f"🛠️  Mode: {args.mode}")
    print("=" * 40)

    reference_text = None
    if args.mode == "speech":
        if not args.reference_speech:
            print("❌ In 'speech' mode, you must provide --reference-speech <file>")
            sys.exit(1)
        try:
            with open(args.reference_speech, "r", encoding="utf-8") as f:
                reference_text = f.read()
        except Exception as e:
            print(f"❌ Could not read reference speech: {e}")
            sys.exit(1)

    # Create and configure speech coach
    coach = SpeechCoach(mode=args.mode, reference_text=reference_text)
    coach.wpm_threshold = args.wpm_threshold
    coach.volume_threshold = args.volume_threshold
    coach.pitch_variation_threshold = args.pitch_threshold

    try:
        coach.start()
    except KeyboardInterrupt:
        print("\n🛑 Session ended by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        coach.stop()


if __name__ == "__main__":
    main()
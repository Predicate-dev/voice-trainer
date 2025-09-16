#!/usr/bin/env python3
"""
Test script for Speech Coach - Simulates speech input for testing purposes.
"""

import time
import sys
from speech_coach import SpeechCoach


def simulate_speech_input():
    """Simulate speech input for testing the WPM calculation."""
    print("🧪 Running Speech Coach simulation...")
    print("This will simulate fast speech to trigger WPM feedback.")
    print("=" * 50)
    
    coach = SpeechCoach()
    
    # Simulate rapid speech input by manually adding word timestamps
    current_time = time.time()
    
    # Simulate speaking 200 words in 60 seconds (200 WPM - above threshold)
    words_per_batch = 10
    batches = 20
    time_between_batches = 3  # 3 seconds between batches
    
    print(f"📊 Simulating {words_per_batch * batches} words over {batches * time_between_batches} seconds")
    print(f"📊 Expected WPM: {(words_per_batch * batches) / (batches * time_between_batches / 60):.0f}")
    print("=" * 50)
    
    for i in range(batches):
        # Add word timestamp
        timestamp = current_time + (i * time_between_batches)
        coach.word_timestamps.append((timestamp, words_per_batch))
        
        # Simulate current time
        time.time = lambda: timestamp
        
        # Test pacing analysis
        coach._analyze_pacing(timestamp)
        
        # Print current state
        if len(coach.word_timestamps) >= 2:
            recent_words = list(coach.word_timestamps)
            total_words = sum(word_count for _, word_count in recent_words)
            time_span = recent_words[-1][0] - recent_words[0][0]
            if time_span > 0:
                wpm = (total_words / time_span) * 60
                print(f"📊 Batch {i+1}: Current WPM = {wpm:.0f}")
        
        # Small delay for readability
        import time as real_time
        real_time.sleep(0.5)
    
    print("\n✅ Simulation completed!")
    print("💡 The coach should have provided 'slow down' feedback when WPM exceeded the threshold.")


def test_basic_functionality():
    """Test basic functionality without microphone."""
    print("🧪 Testing basic Speech Coach functionality...")
    
    try:
        coach = SpeechCoach()
        print("✅ SpeechCoach instance created successfully")
        
        # Test thresholds configuration
        coach.wpm_threshold = 150
        coach.volume_threshold = 0.02
        coach.pitch_variation_threshold = 0.3
        print("✅ Threshold configuration successful")
        
        # Test feedback method
        coach._provide_feedback("🧪 Test feedback message")
        print("✅ Feedback method working")
        
        print("\n📋 Current configuration:")
        print(f"   WPM Threshold: {coach.wpm_threshold}")
        print(f"   Volume Threshold: {coach.volume_threshold}")
        print(f"   Pitch Threshold: {coach.pitch_variation_threshold}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    print("🎤 Speech Coach Test Suite")
    print("=" * 40)
    
    # Test basic functionality
    if test_basic_functionality():
        print("\n" + "=" * 40)
        # Run simulation
        simulate_speech_input()
    else:
        print("❌ Basic functionality test failed")
        sys.exit(1)
    
    print("\n🎯 All tests completed!")
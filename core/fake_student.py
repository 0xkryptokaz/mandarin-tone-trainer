from gtts import gTTS
import librosa
import soundfile as sf
import os

class FakeStudentGenerator:
    """
    Generates synthetic student audio with different types of pronunciation errors.
    Supports multiple TTS engines and error modes for comprehensive testing.
    """
    
    # Available TTS engines
    TTS_ENGINES = {
        'gtts': 'zh-CN',      # Google TTS (default)
        'gtts-tw': 'zh-TW',   # Taiwan accent
        'gtts-en': 'en',      # English reading Chinese (heavy accent)
    }
    
    # Error simulation modes
    ERROR_MODES = {
        'perfect': {'pitch_shift': 0, 'description': 'Perfect pronunciation (baseline)'},
        'tone_high': {'pitch_shift': 2, 'description': 'Tones too high (common beginner error)'},
        'tone_low': {'pitch_shift': -2, 'description': 'Tones too low'},
        'tone_slight_high': {'pitch_shift': 0.5, 'description': 'Slightly high tones'},
        'tone_slight_low': {'pitch_shift': -0.5, 'description': 'Slightly low tones'},
        'tone_very_wrong': {'pitch_shift': 4, 'description': 'Very wrong tones'},
        'english_accent': {'pitch_shift': 0, 'tts_engine': 'gtts-en', 'description': 'Heavy English accent'},
    }
    
    @staticmethod
    def generate(chinese_text, 
                 filename="student_attempt.wav", 
                 error_mode='perfect',
                 tts_engine='gtts',
                 custom_pitch_shift=None):
        """
        Generate synthetic student audio with configurable errors.
        
        Args:
            chinese_text: str - Chinese characters (e.g., "你好")
            filename: str - Output audio file path
            error_mode: str - One of ERROR_MODES keys (e.g., 'tone_high', 'perfect')
            tts_engine: str - One of TTS_ENGINES keys (e.g., 'gtts', 'gtts-tw')
            custom_pitch_shift: float - Override pitch shift (ignores error_mode if set)
        
        Returns:
            dict - Metadata about the generated audio
        """
        
        # Validate inputs
        if error_mode not in FakeStudentGenerator.ERROR_MODES:
            raise ValueError(f"Invalid error_mode. Choose from: {list(FakeStudentGenerator.ERROR_MODES.keys())}")
        
        if tts_engine not in FakeStudentGenerator.TTS_ENGINES:
            raise ValueError(f"Invalid tts_engine. Choose from: {list(FakeStudentGenerator.TTS_ENGINES.keys())}")
        
        # Get error mode configuration
        error_config = FakeStudentGenerator.ERROR_MODES[error_mode].copy()
        
        # Override TTS engine if specified in error mode
        if 'tts_engine' in error_config:
            tts_engine = error_config['tts_engine']
        
        # Override pitch shift if custom value provided
        pitch_shift = custom_pitch_shift if custom_pitch_shift is not None else error_config['pitch_shift']
        
        # Get TTS language code
        lang_code = FakeStudentGenerator.TTS_ENGINES[tts_engine]
        
        # Generate audio
        temp_file = "temp_tts_output.wav"
        try:
            # Generate with selected TTS
            tts = gTTS(text=chinese_text, lang=lang_code)
            tts.save(temp_file)
            
            # Load audio
            y, sr = librosa.load(temp_file, sr=22050)
            
            # Apply pitch shift if needed
            if pitch_shift != 0:
                y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)
            else:
                y_shifted = y
            
            # Save final audio
            sf.write(filename, y_shifted, sr)
            
            # Generate report
            metadata = {
                'filename': filename,
                'chinese_text': chinese_text,
                'error_mode': error_mode,
                'tts_engine': tts_engine,
                'pitch_shift_semitones': pitch_shift,
                'description': error_config['description']
            }
            
            # Print report
            print(f"🤖 Generated synthetic student audio:")
            print(f"   Mode: {error_mode} ({error_config['description']})")
            print(f"   Engine: {tts_engine} (lang={lang_code})")
            print(f"   Pitch shift: {pitch_shift:+.1f} semitones")
            print(f"   Saved to: {filename}")
            
            return metadata
            
        finally:
            # Cleanup
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    @staticmethod
    def list_modes():
        """Print all available error modes and TTS engines."""
        print("\n📋 Available Error Modes:")
        for mode, config in FakeStudentGenerator.ERROR_MODES.items():
            print(f"   '{mode}': {config['description']}")
        
        print("\n🌐 Available TTS Engines:")
        for engine, lang in FakeStudentGenerator.TTS_ENGINES.items():
            print(f"   '{engine}': {lang}")


# Backward compatibility: Keep old function name
def generate_english_accent_pinyin(chinese_text, filename="student_attempt.wav", 
                                   pitch_shift_semitones=2):
    """
    DEPRECATED: Use FakeStudentGenerator.generate() instead.
    Kept for backward compatibility with existing code.
    """
    return FakeStudentGenerator.generate(
        chinese_text=chinese_text,
        filename=filename,
        custom_pitch_shift=pitch_shift_semitones
    )


# Example usage
if __name__ == "__main__":
    # Show all available options
    FakeStudentGenerator.list_modes()
    
    print("\n" + "="*60)
    print("EXAMPLES:")
    print("="*60)
    
    # Example 1: Perfect pronunciation
    FakeStudentGenerator.generate(
        chinese_text="你好",
        filename="test_perfect.wav",
        error_mode='perfect'
    )
    
    # Example 2: High tones error
    FakeStudentGenerator.generate(
        chinese_text="你好",
        filename="test_high_tones.wav",
        error_mode='tone_high'
    )
    
    # Example 3: English accent
    FakeStudentGenerator.generate(
        chinese_text="你好",
        filename="test_english_accent.wav",
        error_mode='english_accent'
    )
    
    # Example 4: Custom pitch shift
    FakeStudentGenerator.generate(
        chinese_text="你好",
        filename="test_custom.wav",
        error_mode='perfect',
        custom_pitch_shift=3.5
    )

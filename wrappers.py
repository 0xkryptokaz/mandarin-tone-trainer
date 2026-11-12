import whisper
import warnings
import librosa

class ASRWrapper:
    def __init__(self, model_size="base"):
        print(f"Loading Whisper model: {model_size}...")
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio_path):
        """
        Transcribes audio with a heavy defensive shield against PyTorch tensor crashes.
        """
        print(f"🎙️ Transcribing {audio_path}...")
        
        # Check if audio is completely broken/empty first
        try:
            duration = librosa.get_duration(path=audio_path)
            if duration < 0.5:
                print("⚠️ Audio too short. Returning empty.")
                return {"text": "", "segments": []}
        except Exception:
            return {"text": "", "segments": []}

        # Attempt 1: Word Timestamps (The one that crashes on dead air)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # This is where PyTorch usually crashes if the mic just picked up static
                result = self.model.transcribe(audio_path, language="zh", word_timestamps=True)
            
            return {
                "text": result["text"],
                "segments": result.get("segments", [])
            }
            
        except Exception as e:
            print(f"⚠️ PyTorch/Whisper crash caught: {e}")
            print("🔄 Falling back to standard transcription (no word timestamps)...")
            
            # Attempt 2: Safe Mode (No timestamps)
            try:
                result_safe = self.model.transcribe(audio_path, language="zh", word_timestamps=False)
                
                # We have to fake a segment so `grade_lesson` doesn't crash expecting an array
                fake_segment = {
                    "start": 0.0,
                    "end": duration,
                    "text": result_safe["text"]
                }
                
                return {
                    "text": result_safe["text"],
                    "segments": [fake_segment] if result_safe["text"] else []
                }
            except Exception as e2:
                print(f"❌ Total Whisper Failure: {e2}")
                return {"text": "", "segments": []}
    def extract_syllable_intervals(self, transcription):
        """Return a list of (start, end) seconds, one per Chinese character."""
        intervals = []
        for seg in transcription.get("segments", []):
            words = seg.get("words")
            if words:
                for w in words:
                    chars = [c for c in w.get("word", "") if "\u4e00" <= c <= "\u9fff"]
                    if not chars:
                        continue
                    step = (w["end"] - w["start"]) / len(chars)
                    for i in range(len(chars)):
                        intervals.append((w["start"] + i * step, w["start"] + (i + 1) * step))
            else:
                # Safe-mode fallback segment: no word timing, so use the whole span
                intervals.append((seg["start"], seg["end"]))
        return intervals

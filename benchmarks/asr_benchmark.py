# benchmarks/asr_benchmark.py
import time
from faster_whisper import WhisperModel
import whisper

def benchmark_faster_whisper(audio_path, model_size="base"):
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    t0 = time.perf_counter()
    segments, info = model.transcribe(audio_path, language="zh", beam_size=5)
    # Consume the generator to force full transcription
    text = "".join(s.text for s in segments)
    return time.perf_counter() - t0

def benchmark_openai_whisper(audio_path, model_size="base"):
    model = whisper.load_model(model_size)
    t0 = time.perf_counter()
    result = model.transcribe(audio_path, language="zh")
    return time.perf_counter() - t0

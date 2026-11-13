#!/usr/bin/env python3
"""
Minimal ASR latency benchmark: OpenAI Whisper vs faster-whisper.

Usage:
    python benchmarks/bench_asr.py --audio path/to/recording.wav
"""
import argparse
import statistics
import time
from pathlib import Path


def bench_openai_whisper(audio_path: str, model_size: str, n_runs: int = 3) -> list[float]:
    import whisper

    model = whisper.load_model(model_size)  # load once
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.transcribe(audio_path, language="zh", word_timestamps=False)
        times.append(time.perf_counter() - t0)
    return times


def bench_faster_whisper(
    audio_path: str,
    model_size: str,
    compute_type: str = "int8",
    n_runs: int = 3,
) -> list[float]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        segments, info = model.transcribe(
            audio_path, language="zh", beam_size=5
        )
        # Force the lazy generator to actually decode everything
        text = "".join(seg.text for seg in segments)
        times.append(time.perf_counter() - t0)
    return times


def summarize(label: str, times: list[float]) -> None:
    print(f"\n{label}")
    print(f"  runs:   {len(times)}")
    print(f"  median: {statistics.median(times)*1000:.1f} ms")
    print(f"  mean:   {statistics.mean(times)*1000:.1f} ms")
    print(f"  min:    {min(times)*1000:.1f} ms")
    print(f"  max:    {max(times)*1000:.1f} ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to a WAV/MP3/FLAC file.")
    parser.add_argument("--model", default="base", help="Whisper model size.")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--faster-compute-type",
        default="int8",
        choices=["int8", "float32", "int8_float32"],
        help="faster-whisper compute_type on CPU.",
    )
    args = parser.parse_args()

    audio = Path(args.audio)
    if not audio.exists():
        raise SystemExit(f"File not found: {audio}")

    print(f"Benchmarking on: {audio} (model={args.model})")

    # Warm up the filesystem / OS cache with a single pass (optional)
    try:
        bench_openai_whisper(str(audio), args.model, n_runs=1)
    except Exception:
        pass

    ow_times = bench_openai_whisper(str(audio), args.model, n_runs=args.runs)
    fw_times = bench_faster_whisper(
        str(audio), args.model, compute_type=args.faster_compute_type, n_runs=args.runs
    )

    summarize("OpenAI Whisper (fp32)", ow_times)
    summarize(f"faster-whisper ({args.faster_compute_type})", fw_times)

    speedup = statistics.median(ow_times) / statistics.median(fw_times)
    print(f"\n→ faster-whisper is {speedup:.1f}x faster (median)")


if __name__ == "__main__":
    main()

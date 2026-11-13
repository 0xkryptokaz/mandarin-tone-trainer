import shutil
import urllib.request
from pathlib import Path

import librosa
import numpy as np
import sherpa_onnx
import soundfile as sf

_MODELS = {
    "sensevoice": {
        "dir_name": "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
        "model_url": "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/model.int8.onnx",
        "tokens_url": "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/tokens.txt",
    },
    "paraformer": {
        "dir_name": "sherpa-onnx-paraformer-zh-2024-03-09",
        "model_url": "https://huggingface.co/csukuangfj/sherpa-onnx-paraformer-zh-2024-03-09/resolve/main/model.int8.onnx",
        "tokens_url": "https://huggingface.co/csukuangfj/sherpa-onnx-paraformer-zh-2024-03-09/resolve/main/tokens.txt",
    },
}


def _download_atomic(url: str, dest_path: Path) -> None:
    """Stream download to a .part file and rename only after full completion."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_suffix(dest_path.suffix + ".part")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(part_path, "wb") as out:
            expected_len = resp.headers.get("Content-Length")
            shutil.copyfileobj(resp, out)

        if expected_len and part_path.stat().st_size != int(expected_len):
            raise IOError(
                f"Incomplete download for {dest_path.name}: "
                f"got {part_path.stat().st_size} of {expected_len} bytes."
            )

        part_path.replace(dest_path)
    except Exception:
        if part_path.is_file():
            part_path.unlink()
        raise


def _ensure_model_files(engine: str) -> tuple[str, str]:
    """Ensure quantized ONNX model and tokens.txt exist in ~/.cache/sherpa-onnx."""
    if engine not in _MODELS:
        raise ValueError(f"Unsupported engine '{engine}'. Choose from: {list(_MODELS)}")

    spec = _MODELS[engine]
    model_dir = Path.home() / ".cache" / "sherpa-onnx" / spec["dir_name"]
    model_path = model_dir / "model.int8.onnx"
    tokens_path = model_dir / "tokens.txt"

    if not model_path.is_file() or model_path.stat().st_size == 0:
        _download_atomic(spec["model_url"], model_path)

    if not tokens_path.is_file() or tokens_path.stat().st_size == 0:
        _download_atomic(spec["tokens_url"], tokens_path)

    return str(model_path), str(tokens_path)


class ASRWrapper:
    def __init__(
        self,
        model_size: str = "sensevoice",
        num_threads: int = 2,
    ):
        engine = "paraformer" if "paraformer" in model_size.lower() else "sensevoice"
        model_path, tokens_path = _ensure_model_files(engine)

        if engine == "paraformer":
            self.recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
                paraformer=model_path,
                tokens=tokens_path,
                num_threads=num_threads,
                debug=False,
            )
        else:
            self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=model_path,
                tokens=tokens_path,
                num_threads=num_threads,
                use_itn=True,
                language="zh",
                debug=False,
            )

    def _load_audio_array(
        self, audio: str | np.ndarray, sample_rate: int = 16000
    ) -> np.ndarray:
        if isinstance(audio, np.ndarray):
            arr = np.asarray(audio, dtype=np.float32)
            if arr.ndim > 1:
                arr = np.mean(arr, axis=1)
            return np.ascontiguousarray(arr, dtype=np.float32)

        data, sr = sf.read(audio, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != sample_rate:
            data = librosa.resample(data, orig_sr=sr, target_sr=sample_rate)
        return np.ascontiguousarray(data, dtype=np.float32)

    def transcribe(self, audio: str | np.ndarray, prompt_hint: str = "") -> dict:
        """Transcribe Mandarin audio purely from acoustics."""
        samples = self._load_audio_array(audio, sample_rate=16000)
        if samples.size == 0:
            return {"text": "", "segments": []}

        stream = self.recognizer.create_stream()
        stream.accept_waveform(16000, samples)
        self.recognizer.decode_stream(stream)

        result = stream.result
        text = result.text.strip()
        duration = float(samples.size) / 16000.0

        words = []
        tokens = getattr(result, "tokens", None) or []
        timestamps = getattr(result, "timestamps", None) or []
        for idx, (tok, t_start) in enumerate(zip(tokens, timestamps)):
            t_end = timestamps[idx + 1] if idx + 1 < len(timestamps) else duration
            words.append({"word": tok, "start": float(t_start), "end": float(t_end)})

        segments = (
            [{"start": 0.0, "end": duration, "text": text, "words": words}]
            if text
            else []
        )
        return {"text": text, "segments": segments}

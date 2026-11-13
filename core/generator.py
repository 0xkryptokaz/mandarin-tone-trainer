import os
import tempfile
from pathlib import Path

import librosa
import soundfile as sf
from gtts import gTTS


def generate_reference_audio(
    text: str,
    filename: str = "teacher_ref.wav",
    target_sr: int = 16000,
) -> str:
    """Generate Mandarin TTS reference audio and save as 16 kHz mono PCM WAV."""
    if not text or not text.strip():
        raise ValueError("Cannot generate reference audio for empty text.")

    out_path = Path(filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write TTS output to a unique temporary MP3 file
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        temp_mp3 = tmp.name

    try:
        tts = gTTS(text=text.strip(), lang="zh-CN")
        tts.save(temp_mp3)

        # Decode MP3, resample to 16 kHz mono, and export as 16-bit PCM WAV
        y, _ = librosa.load(temp_mp3, sr=target_sr, mono=True)
        sf.write(str(out_path), y, target_sr, subtype="PCM_16")
    finally:
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)

    return str(out_path)


if __name__ == "__main__":
    data = generate_reference_audio("你好不客气")
    print(f"Saved test audio to: {data}")

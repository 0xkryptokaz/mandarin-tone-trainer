from pypinyin import pinyin, Style

def analyze_phrase(text: str) -> list[dict]:
    """Analyzes characters, pinyin, and detects Mandarin tone sandhi rules."""
    chars = [c for c in text if "\u4e00" <= c <= "\u9fff"]
    if not chars:
        return []
        
    raw_tones = [p[0] for p in pinyin(chars, style=Style.TONE3, neutral_tone_with_five=True)]
    display_py = [p[0] for p in pinyin(chars, style=Style.TONE)]

    results = []
    for i, char in enumerate(chars):
        # Extract the trailing number from the tone3 format
        orig_tone = int(raw_tones[i][-1]) if raw_tones[i][-1].isdigit() else 5
        spoken_tone = orig_tone
        sandhi_note = None

        # Rule 1: Third-tone sandhi (3 + 3 -> 2 + 3)
        if orig_tone == 3 and i + 1 < len(chars):
            next_tone = int(raw_tones[i + 1][-1]) if raw_tones[i + 1][-1].isdigit() else 5
            if next_tone == 3:
                spoken_tone = 2
                sandhi_note = "Tone 3 → 2"
        
        # Rule 2: "不" (bù) sandhi (4 + 4 -> 2 + 4)
        elif char == "不" and i + 1 < len(chars):
            next_tone = int(raw_tones[i + 1][-1]) if raw_tones[i + 1][-1].isdigit() else 5
            if next_tone == 4:
                spoken_tone = 2
                sandhi_note = "bù → bú"

        results.append({
            "char": char,
            "pinyin": display_py[i],
            "expected_tone": spoken_tone,
            "sandhi": sandhi_note,
        })
    return results

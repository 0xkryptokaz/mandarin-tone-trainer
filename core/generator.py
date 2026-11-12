import json
import shutil
from gtts import gTTS
from pypinyin import pinyin, Style
from pydub import AudioSegment

# CHECK DEPENDENCY: Fail fast if ffmpeg is missing
if not shutil.which("ffmpeg"):
    raise EnvironmentError("FFmpeg is not installed/found! Pydub cannot process audio.")

def apply_comprehensive_sandhi(pinyin_list, chars):
    """
    A slightly smarter Sandhi engine.
    Handles 3-3 rule, plus basic 'Yi' (一) and 'Bu' (不) rules.
    """
    new_list = list(pinyin_list) # Copy list to avoid mutation issues
    
    for i in range(len(new_list)):
        current = new_list[i]
        char = chars[i] # accurate character reference
        
        # Guard clause for last element
        if i + 1 >= len(new_list):
            break
            
        next_p = new_list[i+1]
        
        # RULE 1: 3rd Tone Sandhi (3 + 3 -> 2 + 3)
        if current.endswith('3') and next_p.endswith('3'):
            new_list[i] = current[:-1] + '2'

        # RULE 2: "Bu" (不 - bu4) -> becomes bu2 before tone 4
        if char == '不' and next_p.endswith('4'):
            new_list[i] = 'bu2'

        # RULE 3: "Yi" (一 - yi1) -> rules are complex, simplified here:
        # yi1 + tone 4 -> yi2
        # yi1 + tone 1/2/3 -> yi4
        if char == '一':
            if next_p.endswith('4'):
                 new_list[i] = 'yi2'
            elif any(next_p.endswith(t) for t in ['1', '2', '3']):
                 new_list[i] = 'yi4'

    return new_list

def generate_reference_audio(text, filename="target_audio.wav"):
    # 1. Get raw pinyin
    raw_pinyin_objects = pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)
    raw_pinyin_list = [item[0] for item in raw_pinyin_objects]
    
    # 2. Apply Logic
    final_pinyin = apply_comprehensive_sandhi(raw_pinyin_list, text)

    # 3. Generate Audio
    print(f"Generatng audio for: {text} -> {final_pinyin}")
    tts = gTTS(text, lang='zh-cn')
    
    # Save as MP3 first (gTTS limitation)
    temp_mp3 = "temp_gen.mp3"
    tts.save(temp_mp3)
    
    # 4. Convert using Pydub (Requires FFmpeg)
    sound = AudioSegment.from_mp3(temp_mp3)
    sound = sound.set_frame_rate(16000).set_channels(1)
    sound.export(filename, format="wav")
    
    return {
        "text": text,
        "original_pinyin": raw_pinyin_list,
        "sandhi_pinyin": final_pinyin,
        "audio_path": filename
    }

# --- TEST IT ---
# Try a complex phrase: "Buy one" (Mai3 Yi1) -> Yi changes
# Try "Is not" (Bu4 Shi4) -> Bu changes
data = generate_reference_audio("你好不客气") 
print(json.dumps(data, indent=2, ensure_ascii=False))

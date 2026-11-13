import numpy as np
import sounddevice as sd
from pathlib import Path

from src.pocket_tts import PocketTTS

tts = PocketTTS(
    model_path=Path("checkpoints/pocket-tts-7m-refl")
    )

texts = [
    "Hello there! How was your day?",
    "Anything I can help you with?",
    "That is awesome, keep it up!",
    "I hope you didn't spend $100?"
]

with sd.OutputStream(samplerate=24_000, channels=1, dtype='float32') as stream:
    for text in texts:
        print(f"Generating: \"{text}\"")
        for chunk in tts.stream(text, voice='giovanni'):
            stream.write(chunk)
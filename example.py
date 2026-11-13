import numpy as np
import sounddevice

from pathlib import Path

from src.pocket_tts import PocketTTS

tts = PocketTTS(
    Path("checkpoints/pocket-tts-7m-refl")
)

texts = [
    "Hello there! How was your day?",
    "Anything I can help you with?",
    "That is awesome, keep it up!"
]

print("Generating...")
for text in texts:
    generator = tts.stream(text, voice='giovanni')

    full_audio = np.concat(list(generator), axis=0)

    sounddevice.play(
        data=full_audio, 
        samplerate=24_000, 
        blocking=True
        )
    
import sounddevice
import time
import numpy as np

from pathlib import Path
from time import time_ns

from src.pocket_tts import PocketTTS

tts = PocketTTS(
    Path("checkpoints/pocket-tts-7m-refl")
)

text = "Hi there! How was your day? Anything I can help you with?"

generator = tts.stream(text, voice='alba')
full_audio = np.concat(list(generator), axis=0)

sounddevice.play(full_audio, samplerate=24000, blocking=True)
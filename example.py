import sounddevice
import numpy as np
import matplotlib.pyplot as plt 

from pathlib import Path
from time import time_ns

from src.pocket_tts import PocketTTS

tts = PocketTTS(
    Path("checkpoints/pocket-tts-7m-refl")
)

text = "Hi there! How was your day? anything I can help you with?"

generator = tts.stream(text, voice='alba')
full_audio = np.concat(list(generator), axis=0)


plt.plot(full_audio)
plt.show()

sounddevice.play(full_audio, samplerate=24000, blocking=True)
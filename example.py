import sounddevice as sd

from src.pocket_tts import PocketTTS

tts = PocketTTS(
    model_path="checkpoints/pocket-tts-7m-refl"
    )

texts = [
    "Hello there! How was your day?",
    "Anything I can help you with?",
    "That is awesome, keep it up!",
    "I hope you didn't spend $100?",
    "Dr. Smith bought 5kg of apples for $100.50 when it was 72°F outside.",
    "He did this after visiting https://example.com/shop and emailing support@example.com & saving 50%!"
]

with sd.OutputStream(samplerate=24_000, channels=1, dtype='float32') as stream:
    for text in texts:
        print(f"Generating: \"{text}\"")
        for chunk in tts.stream(text, voice='giovanni'):
            stream.write(chunk)
# Pocket TTS ONNX

A lightweight ONNX port of [Kyutai's Pocket TTS](https://github.com/kyutai-labs/pocket-tts), featuring a CPU-optimized runtime, Q8 quantization and a simple API for streaming generated speech.

## Installation

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

See [`example.py`](example.py) for a minimal working example.

```python
import sounddevice as sd
from src.pocket_tts import PocketTTS

tts = PocketTTS(model_path="checkpoints/pocket-tts-7m-refl")

with sd.OutputStream(samplerate=24_000, channels=1, dtype="float32") as stream:
    for chunk in tts.stream("Hello there! How are you?", voice="giovanni"):
        stream.write(chunk)
```

The `stream()` method yields audio chunks as they are generated, allowing speech to be played back with minimal latency.

Different voices can be selected using the `voice` argument of the `.stream()` method. Pre-compiled voices can be found in the directory `checkpoints/<name_of_model>/voices/`.

## Credits & License

Based on [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) by **Kyutai Labs**.

The original model weights are distributed under **CC BY 4.0**. See the [upstream model card](https://huggingface.co/kyutai/pocket-tts) for licensing and attribution requirements.

This project is an independent ONNX port and quantization effort. Please refer to the upstream project for the original implementation and applicable usage terms.
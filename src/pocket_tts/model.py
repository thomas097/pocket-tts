
import numpy as np
import sentencepiece as spm
from pathlib import Path
from typing import Optional

from tokenizer import Tokenizer
from onnx_session import OnnxSession

class PocketTTS:
    def __init__(self, model_path: Path, quant: Optional[str] = None) -> None:
        # Add suffix to filename on loading a quantized model
        suffix = f'_{quant}' if quant is not None else ""

        self._tokenizer = Tokenizer(
            model_path=model_path / "tokenizer.model"
            )

        self._text_conditioner = OnnxSession(
            model_path=model_path / f"text_conditioner{suffix}.onnx"
            )

    def _encode_text(self, text: str) -> np.typing.NDArray[np.float32]:
        token_ids = self._tokenizer.tokenize(text)
        embeddings, = self._text_conditioner.run(token_ids=token_ids)
        return embeddings


if __name__ == '__main__':
    tts = PocketTTS(
        model_path=Path("checkpoints/pocket-tts-7m"),
        quant="int8"
        )

    result = tts._encode_text("Hello how are you?")
    print(result.shape)

import numpy as np
import sentencepiece as spm
from pathlib import Path
from numpy.typing import NDArray

class Tokenizer:
    def __init__(self, model_path: Path) -> None:
        self._tokenizer = spm.SentencePieceProcessor(
            model_file=str(model_path)
        )

    def tokenize(self, text: str) -> NDArray[np.int64]:
        token_ids = self._tokenizer.encode(text, out_type=int)
        return np.array([token_ids], dtype=np.int64)
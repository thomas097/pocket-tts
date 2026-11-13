import json
import time
from pathlib import Path
from typing import Generator, Optional, Union, Sequence

import numpy as np
import safetensors

from .tokenizer import Tokenizer
from .onnx_session import OnnxSession


_NUMPY_DTYPES: dict = {
    "float32": np.float32,
    "float16": np.float16,
    "int64": np.int64,
    "bool": np.bool_,
}

class PocketTTS:
    VALID_PRECISIONS = ("int8", None)
    TOKENS_PER_SECOND_ESTIMATE = 3.0
    GEN_SECONDS_PADDING = 2.0

    def __init__(
        self,
        model_path: Path,
        voice_dir: str = "voices",
        precision: str = "int8",
        ) -> None:
        if precision not in self.VALID_PRECISIONS:
            raise ValueError(f"precision must be one of {self.VALID_PRECISIONS}, got '{precision}'")

        self._config = self._load_metadata_config(model_path)
        self.voices_path = model_path / voice_dir
        self.precision = precision

        self.frame_rate = float(self._config["frame_rate"])
        self.samples_per_frame = int(self._config["samples_per_frame"])
        self.frame_duration = self.samples_per_frame / self._config["sample_rate"]
        self.latent_dim = int(self._config["latent_dim"])
        self.conditioning_dim = int(self._config["conditioning_dim"])
        
        self.tokenizer = Tokenizer(model_path / "tokenizer.model")

        self.flow_state_manifest = self._config["flow_lm_state_manifest"]
        self.mimi_state_manifest = self._config["mimi_state_manifest"]

        # Load models from (optionally quantized) ONNX files
        suffix: str = f"_{precision}" if precision else ""
        self.mimi_encoder = (
            model_path / f"mimi_encoder{suffix}.onnx"
        )
        self.text_conditioner = OnnxSession(
            model_path / f"text_conditioner{suffix}.onnx"
        )
        self.flow_lm_main = OnnxSession(
            model_path / f"flow_lm_main{suffix}.onnx"
        )
        self.flow_lm_flow = OnnxSession(
            model_path / f"flow_lm_flow{suffix}.onnx"
        )
        self.mimi_decoder = OnnxSession(
            model_path / f"mimi_decoder{suffix}.onnx"
        )
        
        self._voice_cache: dict[str, np.ndarray] = {}
        self._voice_state_cache: dict[str, dict[str, np.ndarray]] = {}

    @staticmethod
    def _load_metadata_config(bundle_dir: Path) -> dict:
        metadata_path = bundle_dir / "config.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing bundle metadata: {metadata_path}")
        return json.loads(metadata_path.read_text())

    def _make_filled_array(self, shape: list[int], dtype, fill: str) -> np.ndarray:
        if fill == "nan":
            return np.full(shape, np.nan, dtype=dtype)
        if fill == "ones":
            return np.ones(shape, dtype=dtype)
        return np.zeros(shape, dtype=dtype)

    def _init_state(self, manifest: list[dict]) -> dict[str, np.ndarray]:
        state = {}
        for entry in manifest:
            dtype = _NUMPY_DTYPES.get(entry["dtype"])
            state[entry["input_name"]] = self._make_filled_array(
                entry["shape"], dtype=dtype, fill=entry["fill"]
            )
        return state

    @staticmethod
    def _clone_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {key: value.copy() for key, value in state.items()}

    def _update_state_from_outputs(
        self,
        state: dict[str, np.ndarray],
        result: Sequence[np.ndarray],
        manifest: list[dict],
        output_offset: int,
    ):
        for entry in manifest:
            state[entry["input_name"]] = result[output_offset + entry["index"]]
    
    @staticmethod
    def _import_model_state_file(source: Union[str, Path]) -> dict[str, dict[str, np.ndarray]]:
        result: dict[str, dict[str, np.ndarray]] = {}
        with safetensors.safe_open(str(source), framework="np") as handle: #type:ignore
            for key in handle.keys():
                module_name, tensor_key = key.split("/", 1)
                result.setdefault(module_name, {})
                result[module_name][tensor_key] = handle.get_tensor(key)
        return result

    @staticmethod
    def _derive_step(module_state: dict[str, np.ndarray]) -> np.ndarray:
        if "step" in module_state:
            return np.asarray(module_state["step"], dtype=np.int64).reshape(1)
        if "offset" in module_state and "end_offset" not in module_state:
            return np.asarray(module_state["offset"], dtype=np.int64).reshape(1)
        if "current_end" in module_state:
            return np.array([module_state["current_end"].shape[0]], dtype=np.int64)
        return np.array([0], dtype=np.int64)

    def _adapt_state_tensor(self, source: np.ndarray, entry: dict) -> np.ndarray:
        target_shape = tuple(entry["shape"])
        target_dtype = _NUMPY_DTYPES.get(entry["dtype"])
        source = np.asarray(source, dtype=target_dtype)

        if source.shape == target_shape:
            return source.copy()

        if source.size == np.prod(target_shape, dtype=np.int64):
            return source.reshape(target_shape).copy()

        target = self._make_filled_array(list(target_shape), target_dtype, entry["fill"])
        if source.ndim != len(target_shape):
            return target

        slices = tuple(slice(0, min(src, dst)) for src, dst in zip(source.shape, target_shape))
        if all(s.start == s.stop for s in slices):
            return target
        target[slices] = source[slices]
        return target

    def _state_from_model_state(
        self, model_state: dict[str, dict[str, np.ndarray]], manifest: list[dict]
    ) -> dict[str, np.ndarray]:
        state = self._init_state(manifest)
        for entry in manifest:
            module_state = model_state.get(entry["module"], {})
            tensor = module_state.get(entry["key"])
            if tensor is None and entry["key"] == "step":
                tensor = self._derive_step(module_state)
            if tensor is None:
                continue
            state[entry["input_name"]] = self._adapt_state_tensor(tensor, entry)
        return state

    def prepare_voice_state(self, voice: str) -> dict[str, np.ndarray]:
        if voice in self._voice_state_cache:
            return self._clone_state(self._voice_state_cache[voice])

        voice_path = self.voices_path / f"{voice}.safetensors"
        if voice_path.exists() and voice_path.suffix == ".safetensors":
            model_state = self._import_model_state_file(voice_path)
            voice_state = self._state_from_model_state(model_state, self.flow_state_manifest)

            self._voice_state_cache[voice] = voice_state

            return voice_state
        else:
            raise ValueError(f"Voice '{voice}' not found.")

    def _prepare_text_prompt(self, text: str) -> tuple[str, int]:
        text = text.strip()
        if not text:
            raise ValueError("Text cannot be empty")
        text = text.replace("\n", " ").replace("\r", " ").replace("  ", " ")

        number_of_words = len(text.split())
        frames_after_eos_guess = 3 if number_of_words <= 4 else 1

        if not text[0].isupper():
            text = text[0].upper() + text[1:]
        if text[-1].isalnum():
            text = text + "."
        return text, frames_after_eos_guess

    def _estimate_max_gen_len(self, token_count: int) -> int:
        gen_len_sec = token_count / self.TOKENS_PER_SECOND_ESTIMATE + self.GEN_SECONDS_PADDING
        return int(np.ceil(gen_len_sec * self.frame_rate))

    def _run_flow_lm_chunk(
        self,
        initial_state: dict[str, np.ndarray],
        text_ids: np.ndarray,
        max_frames: Optional[int],
        frames_after_eos: int,
    ) -> Generator[np.ndarray, None, None]:
        state = self._clone_state(initial_state)
        text_embeddings, = self.text_conditioner.run(token_ids=text_ids)
        if text_embeddings.ndim == 2:
            text_embeddings = text_embeddings[None]

        empty_seq = np.zeros((1, 0, self.latent_dim), dtype=np.float32)
        empty_text = np.zeros((1, 0, self.conditioning_dim), dtype=np.float32)

        result = self.flow_lm_main.run(
            sequence=empty_seq, 
            text_embeddings=text_embeddings,
            **state
            )
        self._update_state_from_outputs(state, result, self.flow_state_manifest, output_offset=2)

        curr = np.full((1, 1, self.latent_dim), np.nan, dtype=np.float32)
        eos_step = None
        frame_limit = max_frames or self._estimate_max_gen_len(text_ids.shape[1])
        dt = 1.0

        for step in range(frame_limit):
            result = self.flow_lm_main.run(
                sequence=curr, 
                text_embeddings=empty_text, 
                **state
                )
            conditioning = result[0]
            eos_logit = result[1]
            self._update_state_from_outputs(state, result, self.flow_state_manifest, output_offset=2)

            if eos_logit[0][0] > -4.0 and eos_step is None:
                eos_step = step
            if eos_step is not None and step >= eos_step + frames_after_eos:
                break

            x = np.zeros((1, self.latent_dim), dtype=np.float32)

            flow = self.flow_lm_flow.run(
                c=conditioning,  
                x=x
            )[0]
            x = x + flow * dt

            latent = x.reshape(1, 1, self.latent_dim)
            yield latent
            curr = latent

    def stream(
        self,
        text: str,
        voice: str,
        max_frames: Optional[int] = None,
        frames_after_eos: Optional[int] = None,
        first_chunk_frames: int = 2,
        target_buffer_sec: float = 0.2,
        max_chunk_frames: int = 15,
    ) -> Generator[np.ndarray, None, None]:
        base_state = self.prepare_voice_state(voice)

        for chunk_text in [text]:
            _, guess = self._prepare_text_prompt(chunk_text)
            effective_frames = (
                frames_after_eos if (frames_after_eos is not None) else guess + 2
            )

            prepared, _ = self._prepare_text_prompt(text)
            token_ids = self.tokenizer.tokenize(prepared)

            mimi_state = self._init_state(self.mimi_state_manifest)
            generated_latents = []
            decoded_frames = 0
            playback_start_time = None
            start_time = time.time()

            for latent in self._run_flow_lm_chunk(base_state, token_ids, max_frames, effective_frames):
                generated_latents.append(latent)
                pending = len(generated_latents) - decoded_frames
                chunk_size = 0

                if playback_start_time is None:
                    if pending >= first_chunk_frames:
                        chunk_size = first_chunk_frames
                else:
                    elapsed = time.time() - start_time
                    audio_decoded_sec = decoded_frames * self.frame_duration
                    playback_elapsed = elapsed - playback_start_time
                    buffer_sec = audio_decoded_sec - playback_elapsed

                    if buffer_sec < target_buffer_sec and pending >= 1:
                        chunk_size = min(pending, 3)
                    elif pending >= max_chunk_frames:
                        chunk_size = max_chunk_frames

                if chunk_size > 0:
                    latents_chunk = np.concatenate(
                        generated_latents[decoded_frames : decoded_frames + chunk_size], axis=1
                    )
                    result = self.mimi_decoder.run(latent=latents_chunk, **mimi_state)
                    self._update_state_from_outputs(
                        mimi_state, result, self.mimi_state_manifest, output_offset=1
                    )
                    decoded_frames += chunk_size
                    if playback_start_time is None:
                        playback_start_time = time.time() - start_time
                    yield result[0].reshape(-1)

            if decoded_frames < len(generated_latents):
                latents_chunk = np.concatenate(generated_latents[decoded_frames:], axis=1)
                result = self.mimi_decoder.run(latent=latents_chunk, **mimi_state)[0] # type:ignore
                yield result[0].reshape(-1)
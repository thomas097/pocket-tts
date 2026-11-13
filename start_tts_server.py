import grpc
import yaml
import logging
import numpy as np
from concurrent import futures
from pathlib import Path
from typing import Any

import src.pocket_tts.grpc.tts_pb2 as tts_pb2
import src.pocket_tts.grpc.tts_pb2_grpc as tts_pb2_grpc

from src.pocket_tts import PocketTTS

class TTSServiceServicer(tts_pb2_grpc.TTSServiceServicer):
    def __init__(self, config: dict):
        self.config = config
        tts_config = config.get("tts", {})
        
        checkpoint_path = Path(tts_config.get("checkpoint_path", "checkpoints/pocket-tts-7m-refl"))
        print(f"Loading PocketTTS model from '{checkpoint_path}'...")
        self.tts = PocketTTS(checkpoint_path)
        print("Model loaded successfully.")

    def stream(self, request: tts_pb2.TTSRequest, context: Any):
        tts_config = self.config.get("tts", {})
        text = request.text
        voice = request.voice if request.voice else tts_config.get("default_voice", "giovanni")
        sample_rate = tts_config.get("sample_rate", 24_000)

        print(f"Received request for text: '{text}' with voice '{voice}'")

        try:
            for chunk in self.tts.stream(text, voice=voice):
                if isinstance(chunk, np.ndarray):
                    audio_bytes = chunk.astype(np.float32).tobytes()
                else:
                    audio_bytes = bytes(chunk)

                yield tts_pb2.AudioChunk(
                    audio_data=audio_bytes,
                    sample_rate=sample_rate
                )
        except Exception as e:
            logging.error(f"Error during TTS generation: {e}")
            context.abort(grpc.StatusCode.INTERNAL, str(e))

def load_config(config_path: str = "config.yaml") -> dict:
    """Loads the YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path.absolute()}")
    
    with open(path, mode="r") as f:
        return yaml.safe_load(f)
    
def serve():
    # Load configuration prior to starting server components
    config = load_config("config.yaml")
    server_config = config.get("server", {})
    
    max_workers = server_config.get("max_workers", 10)
    host = server_config.get("host", "[::]")
    port = server_config.get("port", 50051)
    address = f"{host}:{port}"

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    tts_pb2_grpc.add_TTSServiceServicer_to_server(
        TTSServiceServicer(config),
        server
    )
    
    server.add_insecure_port(address)
    print(f"gRPC TTS Server started and listening on {address}...")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    serve()
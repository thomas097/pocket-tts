from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class TTSRequest(_message.Message):
    __slots__ = ("text", "voice")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    VOICE_FIELD_NUMBER: _ClassVar[int]
    text: str
    voice: str
    def __init__(self, text: _Optional[str] = ..., voice: _Optional[str] = ...) -> None: ...

class AudioChunk(_message.Message):
    __slots__ = ("audio_data", "sample_rate")
    AUDIO_DATA_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    audio_data: bytes
    sample_rate: int
    def __init__(self, audio_data: _Optional[bytes] = ..., sample_rate: _Optional[int] = ...) -> None: ...

import onnxruntime as ort
import numpy.typing as npt
from pathlib import Path
from typing import Any, Sequence

class OnnxSession:
    def __init__(self, model_path: Path) -> None:
        self._session = ort.InferenceSession(
            path_or_bytes=model_path,
            sess_options=self._default_session_options(),
            providers=['CPUExecutionProvider']
        )

    def _default_session_options(self) -> ort.SessionOptions:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        return options

    # Public API

    @property
    def inputs(self) -> Sequence[ort.NodeArg]:
        return self._session.get_inputs()

    @property
    def outputs(self) -> Sequence[ort.NodeArg]:
        return self._session.get_outputs()

    def run(self, **inputs) -> Sequence[npt.NDArray[Any]]:
        result = self._session.run(
            output_names=None,
            input_feed=inputs
            )
        return result #type:ignore
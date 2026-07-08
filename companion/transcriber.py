# companion/transcriber.py
import glob
import os
import site

import numpy as np
from faster_whisper import WhisperModel


def _register_pip_nvidia_dll_dirs() -> None:
    # pip's nvidia-* wheels (cuBLAS/cuDNN) put their DLLs in
    # site-packages/nvidia/*/bin, which Windows' DLL loader does not search.
    # Without registering those directories, CTranslate2 fails with
    # "cublas64_12.dll is not found" at the first GPU transcription.
    # No-op on non-Windows platforms and CPU-only installs.
    if not hasattr(os, "add_dll_directory"):
        return
    for root in site.getsitepackages() + [site.getusersitepackages()]:
        for dll_dir in glob.glob(os.path.join(root, "nvidia", "*", "bin")):
            os.add_dll_directory(dll_dir)


class LocalTranscriber:
    def __init__(self, model_size: str, device: str, compute_type: str):
        _register_pip_nvidia_dll_dirs()
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(audio, beam_size=5)
        return " ".join(segment.text.strip() for segment in segments).strip()


def make_transcriber(name: str):
    from companion import config

    if name == "local":
        return LocalTranscriber(
            config.WHISPER_MODEL_SIZE,
            config.WHISPER_DEVICE,
            config.WHISPER_COMPUTE_TYPE,
        )
    raise ValueError(f"Unknown transcriber: {name}")

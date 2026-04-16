

import torch
import numpy as np
from pathlib import Path


class SileroVAD:
    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000,
        device: str = torch.device('cpu'),
    ):
        self.sample_rate = sample_rate
        self.device = device

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Silero VAD not found: {model_path}")

        self.model = torch.jit.load(str(model_path), map_location=device)
        self.model.eval()

        self._reset_states()

    # -------------------------
    # Estado interno
    # -------------------------
    def _reset_states(self):
        self.h = None
        self.c = None

    def reset(self):
        self._reset_states()

    # -------------------------
    # Inferencia
    # -------------------------
    @torch.no_grad()
    def __call__(self, audio: np.ndarray) -> float:
        """
        audio: np.ndarray float32, shape (N,)
        returns: speech probability [0, 1]
        """

        if audio.ndim != 1:
            audio = audio.flatten()

        audio = torch.from_numpy(audio).float().to(self.device)

        if self.sample_rate != 16000:
            raise ValueError("Silero VAD expects 16kHz audio")

        speech_prob = self.model(
            audio, self.sample_rate
        ).item()

        return float(speech_prob)

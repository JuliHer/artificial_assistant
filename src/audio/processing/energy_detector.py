import numpy as np

class EnergyDetector:
    def __init__(
        self,
        k=3.0,
        alpha_fast=0.05,
        alpha_slow=0.005,
        change_ratio=1.8,
        change_frames=6,
    ):
        self.k = k

        self.alpha_fast = alpha_fast
        self.alpha_slow = alpha_slow

        self.noise_fast = None
        self.noise_slow = None

        self.change_ratio = change_ratio
        self.change_frames = change_frames
        self._change_counter = 0

        self._locked = False
        self._initialized = False


    # -------------------------
    # RMS
    # -------------------------
    def rms(self, audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(audio ** 2)))

    # -------------------------
    # Inicialización
    # -------------------------
    def initialize(self, audio: np.ndarray):
        value = self.rms(audio)
        self.noise_fast = value
        self.noise_slow = value
        self._initialized = True

    # -------------------------
    # Actualización noise floor
    # -------------------------
    def update_noise(self, audio: np.ndarray):
        value = self.rms(audio)

        if self.noise_fast is None:
            self.noise_fast = value
            self.noise_slow = value
            return

        self.noise_fast = (
            (1 - self.alpha_fast) * self.noise_fast
            + self.alpha_fast * value
        )

        self.noise_slow = (
            (1 - self.alpha_slow) * self.noise_slow
            + self.alpha_slow * value
        )

    def environment_changed(self) -> bool:
        if self.noise_fast is None or self.noise_slow is None:
            return False

        ratio = self.noise_fast / max(self.noise_slow, 1e-6)

        if ratio > self.change_ratio:
            self._change_counter += 1
        else:
            self._change_counter = 0

        return self._change_counter >= self.change_frames

    # -------------------------
    # Decisión
    # -------------------------
    def is_voice(self, audio: np.ndarray) -> bool:
        if self.noise_slow is None:
            return False

        value = self.rms(audio)
        threshold = self.noise_slow * self.k


        is_voice = value > threshold
        if not is_voice:
            self.update_noise(audio)
            if self.environment_changed():
                self.initialize(audio)

        return is_voice


    # -------------------------
    # Debug
    # -------------------------
    @property
    def noise_floor(self):
        if not self._initialized:
            return None
        return self.noise_slow

    @property
    def threshold(self):
        if not self._initialized:
            return None
        return self.noise_slow * self.k
import numpy as np

class Recorder:
    def __init__(
        self,
        energy,
        silence_ms=600,
        max_duration_ms=6000,
        hop_ms=20,
        min_duration_ms=300,
    ):
        self.energy = energy

        self.silence_frames = int(silence_ms / hop_ms)
        self.max_frames = int(max_duration_ms / hop_ms)
        self.min_frames = int(min_duration_ms / hop_ms)

        self.reset()

    # -------------------------
    # Control
    # -------------------------
    def reset(self):
        self.buffer = []
        self._silence_count = 0
        self._recording = False

    def start(self):
        self.reset()
        self._recording = True

    def append(self, audio: np.ndarray):
        if not self._recording:
            return

        self.buffer.append(audio)

        if self.energy.is_voice(audio):
            self._silence_count = 0
        else:
            self._silence_count += 1

    def should_stop(self) -> bool:
        if not self._recording:
            return False

        frames = len(self.buffer)

        if frames >= self.max_frames:
            return True

        if frames >= self.min_frames and self._silence_count >= self.silence_frames:
            return True

        return False

    def stop(self) -> np.ndarray:
        self._recording = False

        if not self.buffer:
            return np.array([], dtype=np.float32)

        clip = np.concatenate(self.buffer, axis=0)
        self.reset()
        return clip

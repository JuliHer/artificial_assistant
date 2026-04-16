import time


class SoftVAD:
    def __init__(
        self,
        vad,
        on_threshold=0.6,
        off_threshold=0.3,
        min_speech_frames=3,
        hangover_frames=5,
    ):
        """
        vad: callable -> float [0,1]
        """

        assert on_threshold > off_threshold

        self.vad = vad

        self.on_th = on_threshold
        self.off_th = off_threshold

        self.min_speech_frames = min_speech_frames
        self.hangover_frames = hangover_frames

        self._speech_frames = 0
        self._silence_frames = 0
        self._active = False

        self._last_prob = 0.0
        self._last_ts = time.monotonic()

    # -------------------------
    # Estado
    # -------------------------
    def reset(self):
        self._speech_frames = 0
        self._silence_frames = 0
        self._active = False
        self._last_prob = 0.0
        self._last_ts = time.monotonic()

        if hasattr(self.vad, "reset"):
            self.vad.reset()

    # -------------------------
    # Inferencia
    # -------------------------
    def is_speech(self, audio) -> bool:
        p = self.vad(audio)
        self._last_prob = p
        self._last_ts = time.monotonic()

        if self._active:
            # ya estamos en voz
            if p < self.off_th:
                self._silence_frames += 1
                if self._silence_frames >= self.hangover_frames:
                    self._active = False
                    self._speech_frames = 0
                    self._silence_frames = 0
            else:
                self._silence_frames = 0
        else:
            # estamos en silencio
            if p > self.on_th:
                self._speech_frames += 1
                if self._speech_frames >= self.min_speech_frames:
                    self._active = True
                    self._silence_frames = 0
            else:
                self._speech_frames = 0

        return self._active

    # -------------------------
    # Debug
    # -------------------------
    @property
    def active(self):
        return self._active

    @property
    def last_probability(self):
        return self._last_prob

    @property
    def last_timestamp(self):
        return self._last_ts

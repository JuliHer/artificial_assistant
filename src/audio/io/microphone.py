import sounddevice as sd
import numpy as np
import threading

class Microphone:

    def __init__(self, sample_rate=16000, channels=1, buffer_seconds=5, device=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_size = int(sample_rate*buffer_seconds)
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_index = 0

        self.lock = threading.Lock()
        self.stream = None
        self.device = device


    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(status)

        data = indata[:, 0]

        with self.lock:
            for sample in data:
                self.buffer[self.write_index] = sample
                self.write_index = (self.write_index + 1) % self.buffer_size

    def start(self):
        if self.stream is not None:
            return

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._audio_callback,
            device=self.device,
            dtype="float32",
            blocksize=0
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def get_last_seconds(self, seconds=1.0):
        samples = int(seconds * self.sample_rate)

        if samples > self.buffer_size:
            raise ValueError("Más segundos de los que caben en el buffer")

        with self.lock:
            end = self.write_index
            start = (end - samples) % self.buffer_size

            if start < end:
                return self.buffer[start:end].copy()
            else:
                return np.concatenate(
                    (self.buffer[start:], self.buffer[:end])
                ).copy()
import numpy as np
from collections import deque
import librosa
import matplotlib.pyplot as plt
import os
import datetime
from pathlib import Path
import soundfile as sf
import time

class WakeWordController:
    def __init__(
        self,
        model,
        sample_rate=16000,
        n_mel=40,
        frame_ms=32,
        hop_ms=20,
        frames=64,
        threshold=0.8,
    ):
        
        
        self.model = model
        self.sample_rate = sample_rate
        self.n_mel = n_mel
        self.threshold = threshold

        self.frame_len = int(sample_rate * frame_ms / 1000)
        self.hop_len = int(sample_rate * hop_ms / 1000)

        self.frames = frames

        self.buffer = deque(maxlen=self.frames)
        self.buffer_audio = deque(maxlen=self.frames)
        self.latencies = []
        self.ready = False

        #plt.ion()
        #self.fig, self.ax = plt.subplots()

        #self.img = None
        

    # -------------------------
    # Feature extraction
    # -------------------------
    def extract_mel(self, audio: np.ndarray) -> np.ndarray:
        
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.frame_len,
            hop_length=self.frame_len,
            n_mels=self.n_mel
        )

        mel = librosa.power_to_db(mel, ref=1.0)


        return mel

    # -------------------------
    # Main processing
    # -------------------------

    def push(self, audio: np.ndarray):
        self.energy = np.sum(audio ** 2)

        mel = self.extract_mel(audio)

        
        self.buffer.append(mel[:,0])
        self.buffer_audio.append(audio[:self.hop_len])

        self.ready = len(self.buffer) == self.frames

        #if self.ready:
        #    self.plot_live(self.buffer)
            
    def save_window(self, dir):
        
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent
        try:

            os.makedirs(str(project_root / dir ), exist_ok=True)
            filename = f"{datetime.datetime.now().strftime('save_%d%m%y_%H%M%S')}.wav"
            
            
            path = os.path.join(str(project_root / dir), filename)

            audio_to_save = np.concatenate(self.buffer_audio).flatten()

            print(len(audio_to_save))
            if np.max(np.abs(audio_to_save)) > 0:
                audio_to_save = audio_to_save / np.max(np.abs(audio_to_save))
            sf.write(
                path, 
                audio_to_save, 
                int(self.sample_rate)
                )
        except Exception as e:
            print(e)
            return False

        return True

    def plot_live(self, buffer):
        if len(buffer) == 0:
            return

        mel_matrix = np.stack(buffer)
        

        if self.img is None:

            self.img = self.ax.imshow(
                mel_matrix.T,
                aspect="auto",
                origin="lower"
            )

            plt.colorbar(self.img)

        else:

            self.img.set_data(mel_matrix.T)

        if self.energy > 0.01:
                self.ax.axvline(
                    x=len(self.buffer) - 1,
                    linewidth=1
                )
        plt.pause(0.001)

    def predict(self) -> float:
        
        if not self.ready:
            print(" no listo")
            return 0.0
        
        t0 = time.perf_counter()
        
        window = np.stack(self.buffer)  # (T, n_mfcc)

        window = (window - np.mean(window)) / (np.std(window) + 1e-6)

        window = np.expand_dims(window, axis=-1)  # (64, 40, 1)

        # agregar batch
        window = np.expand_dims(window, axis=0)   # (1, 64, 40, 1)
        score = float(self.model.predict(window, verbose=0)[0][0])

        t1 = time.perf_counter()
    
        # Guardar latencia en una lista de la clase (inicialízala en __init__)
        self.latencies.append((t1 - t0) * 1000)
        
        if len(self.latencies) >= 100:
            avg = sum(self.latencies) / 100
            print(f"--- Promedio de 100 inferencias: {avg:.2f} ms ---")
            self.latencies = []
        
        return score

    # -------------------------
    # Utils
    # -------------------------
    def reset(self):
        self.buffer.clear()

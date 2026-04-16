import torch
import numpy as np
import sounddevice as sd

class VoiceDetector:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                          model='silero_vad',
                                          force_reload=False)
        (self.get_speech_timestamps, _, self.read_audio, _, _) = utils
        self.model.eval()

    def is_speech(self, audio_chunk):
        """
        Analiza un trozo de audio y devuelve True si detecta voz humana.
        """
        # Convertir el chunk de audio a tensor de torch
        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        new_confidence = self.model(torch.from_numpy(audio_float32), self.sample_rate).item()
        
        return new_confidence > 0.5  # Umbral de confianza (ajustable)

    def listen_loop(self):
        """Ejemplo de bucle de escucha para el MVP"""
        print("KORA escuchando...")
        with sd.RawInputStream(samplerate=self.sample_rate, blocksize=512, 
                               dtype='int16', channels=1) as stream:
            while True:
                audio_chunk, overflowed = stream.read(512)
                if self.is_speech(audio_chunk):
                    print("¡Voz detectada!")
                    # Aquí dispararías el siguiente paso: 
                    # 1. Grabar hasta el silencio -> 2. STT -> 3. KORA Inference
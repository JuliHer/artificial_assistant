import os
import librosa
import soundfile as sf


class WakeWordAugmenter:

    def __init__(self, sr=16000):

        self.sr = sr

        # configuraciones de augmentación
        self.pitch_steps = [-2, 2]
        self.speed_factors = [0.9, 1.1]
        self.gain_db = [-6, 6]

    def change_pitch(self, audio, steps):

        return librosa.effects.pitch_shift(
            audio,
            sr=self.sr,
            n_steps=steps
        )

    def change_speed(self, audio, factor):

        return librosa.effects.time_stretch(
            audio,
            rate=factor
        )

    def apply_gain(self, audio, db):

        factor = 10 ** (db / 20)
        return audio * factor

    def augment_file(self, input_path, output_dir, index):

        audio, _ = librosa.load(input_path, sr=self.sr)

        audio = librosa.util.normalize(audio)
        os.makedirs(output_dir, exist_ok=True)

        counter = 0

        # guardar original
        out_name = f"sample_kora_{index}_{counter}.wav"
        sf.write(os.path.join(output_dir, out_name), audio, self.sr)
        counter += 1

        # pitch
        for p in self.pitch_steps:

            aug = self.change_pitch(audio, p)

            aug = librosa.util.normalize(aug)
            out_name = f"sample_kora_{index}_{counter}.wav"
            
            sf.write(os.path.join(output_dir, out_name), aug, self.sr)

            counter += 1

        # velocidad
        for s in self.speed_factors:

            aug = self.change_speed(audio, s)
            aug = librosa.util.normalize(aug)

            out_name = f"sample_kora_{index}_{counter}.wav"
            sf.write(os.path.join(output_dir, out_name), aug, self.sr)

            counter += 1

        # ganancia
        for g in self.gain_db:

            aug = self.apply_gain(audio, g)

            aug = librosa.util.normalize(aug)

            out_name = f"sample_kora_{index}_{counter}.wav"
            sf.write(os.path.join(output_dir, out_name), aug, self.sr)

            counter += 1
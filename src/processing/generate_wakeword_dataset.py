import os
import json
import random
import numpy as np
import librosa
import soundfile as sf


SR = 16000


class WakeWordDatasetGenerator:

    def __init__(self, augmenter, wakeword_dir, artificial_noises_dir, natural_noises_dir, output_dir):
        self.augmenter = augmenter
        self.wakeword_dir = wakeword_dir
        self.artificial_noises_dir = artificial_noises_dir
        self.natural_noises_dir = natural_noises_dir
        self.output_dir = output_dir

        self.samples_counter = 0
        self.index = 0

        self.counter = 0
        self.audio_out = os.path.join(output_dir, "audio")
        os.makedirs(self.audio_out, exist_ok=True)

        self.samples = []

    def mix_audio(self, noise, wakeword, position):
        end = position + len(wakeword)

        noise[position:end] += wakeword

        noise = np.clip(noise, -1.0, 1.0)

        return noise
    
    def fix_length(self, audio, target_len):

        if len(audio) > target_len:

            return audio[:target_len]

        if len(audio) < target_len:

            repeats = target_len // len(audio) + 1

            audio = np.tile(audio, repeats)

            return audio[:target_len]

        return audio

    def normalize(self, audio, target_db):

        rms = np.sqrt(np.mean(audio**2))

        scalar = 10 ** (target_db / 20) / (rms + 1e-8)

        return audio * scalar
    
    def generate_augmented_wake(self, wake_dir, output_dir):
        wakewords = os.listdir(wake_dir)

        index = 0

        for wakeword in wakewords:

            if not wakeword.endswith(".wav"):
                continue

            wakeword_path = os.path.join(wake_dir, wakeword)

            self.augmenter.augment_file(wakeword_path, output_dir, index)
            
            index += 1

        print(f"datos aumentados correctamente: {output_dir}")


    def generate_wake_w_noises(self):
        wakewords = os.listdir(self.wakeword_dir)
        noises = os.listdir(self.artificial_noises_dir)

        for wakeword in wakewords:

            if not wakeword.endswith(".wav"):
                continue

            self.index += 1

            wakeword_path = os.path.join(self.wakeword_dir, wakeword)

            wake_file, _ = librosa.load(wakeword_path, sr=SR)
            
            target_noise_db = random.randint(-35, -20)
            target_wake_db = target_noise_db + random.randint(-3, 8)
            
            wake_file = self.normalize(wake_file, target_wake_db)
            wake_len = len(wake_file) / SR
            noises_annotation = []
            sample_name = f"sample_{self.samples_counter}"
            for counter in range(1,7):

                noise = random.choice(noises)

                noise_path = os.path.join(self.artificial_noises_dir, noise)

                noise_file, _ = librosa.load(noise_path, sr=SR)

                noise_file = self.fix_length(noise_file, SR*5)
                target_noise_db
                noise_file = self.normalize(noise_file, target_noise_db)

                margin = int(SR * 0.5)

                start = random.randint(
                    margin,
                    len(noise_file) - margin - len(wake_file)
                )

                mixed = noise_file.copy()

                mixed = self.mix_audio(mixed, wake_file, start)


                mixed_filename = f"sample_{self.index}_{counter}.wav"

                mix_path = os.path.join(
                    self.audio_out,
                    mixed_filename
                )

                sf.write(
                    mix_path,
                    mixed,
                    SR
                )

                start_time = (start / SR)

                end_time = start_time + wake_len

                noises_annotation.append(
                    {
                        "file":
                        mixed_filename,

                        "wake_intervals":[
                            {
                                "start":
                                float(start_time),

                                "end":
                                float(end_time)
                            }
                        ]
                    }
                )
                self.counter += 1

            sample_entry = {
                "sample":
                sample_name,
                "type": 0,
                "files": noises_annotation
            }

            self.samples_counter += 1

            self.samples.append(
                sample_entry
            )

        print("Dataset de wake words con ruido artificial creado correctamente")


    def pass_noises_to_dataset(self):

        noises = os.listdir(self.artificial_noises_dir)
        b = 0
        for noise_file in noises:

            if not noise_file.endswith(".wav"):
                continue

            sample_name = f"sample_{self.samples_counter}"

            noise_path = os.path.join(
                self.artificial_noises_dir,
                noise_file
            )

            noise, _ = librosa.load(
                noise_path,
                sr=SR
            )

            noise_name = f"sample_{self.index}_0.wav"

            original_path = os.path.join(
                self.audio_out,
                noise_name
            )

            sf.write(
                original_path,
                noise,
                SR
            )

            sample_entry = {
                "sample":
                sample_name,
                "type": 0,
                "files": [
                    {
                        "file":
                        noise_name,
                        "wake_intervals":[]
                    }
                ]
            }

            self.samples_counter += 1

            self.index += 1
            self.counter += 1
            self.samples.append(
                sample_entry
            )

            b += 1

        print(f"{b} archivos copiados correctamente")



    def generate_noises_w_wake(self):

        wakewords = os.listdir(self.wakeword_dir)
        noises = os.listdir(self.natural_noises_dir)
        
        for noise_file in noises:

            if not noise_file.endswith(".wav"):
                continue


            self.index += 1

            sample_name = f"sample_{self.samples_counter}"

            noise_path = os.path.join(
                self.natural_noises_dir,
                noise_file
            )

            noise, _ = librosa.load(
                noise_path,
                sr=SR
            )

            # ---------- Guardar noise original ----------
            original_filename = f"sample_{self.index}_0.wav"
            
            original_path = os.path.join(
                self.audio_out,
                original_filename
            )

            sf.write(
                original_path,
                noise,
                SR
            )

            # ---------- Crear mezcla ----------

            noise_len = len(noise) / SR

            ww_min = max(
                1,
                int(noise_len / 20)
            )

            ww_max = max(
                1,
                int(noise_len / 10)
            )

            ww_count = random.randint(
                ww_min,
                ww_max
            )

            mixed = noise.copy()

            wakeword_annotations = []

            used_positions = []


            for _ in range(ww_count):

                ww_file = random.choice(
                    wakewords
                )

                ww_path = os.path.join(
                    self.wakeword_dir,
                    ww_file
                )

                wakeword, _ = librosa.load(
                    ww_path,
                    sr=SR
                )

                ww_len_samples = len(
                    wakeword
                )

                ww_len = (
                    ww_len_samples / SR
                )

                for _ in range(10):

                    position = random.randint(
                        0,
                        len(mixed)
                        - ww_len_samples
                    )

                    valid = True

                    for p in used_positions:

                        if abs(position - p) < SR * 2:

                            valid = False
                            break

                    if valid:

                        used_positions.append(
                            position
                        )

                        break

                mixed = self.mix_audio(
                    mixed,
                    wakeword,
                    position
                )

                start_time = (
                    position / SR
                )

                end_time = (
                    start_time + ww_len
                )

                wakeword_annotations.append({

                    "start":
                    float(start_time),

                    "end":
                    float(end_time)

                })

                

            # ---------- Guardar mix ----------

            mix_filename = f"sample_{self.index}_1.wav"

            self.counter += 1


            mix_path = os.path.join(
                self.audio_out,
                mix_filename
            )

            

            sf.write(
                mix_path,
                mixed,
                SR
            )

            # ---------- Registrar sample ----------

            sample_entry = {

                "sample":
                sample_name,
                "type": 1,
                "files": [

                    {
                        "file":
                        original_filename,

                        "wake_intervals":
                        []

                    },

                    {
                        "file":
                        mix_filename,

                        "wake_intervals":
                        wakeword_annotations

                    }

                ]

            }

            self.samples.append(
                sample_entry
            )

            self.samples_counter += 1

        print("Dataset de ruido natural con wake words correctamente")

    def save_annotations(self):

        dataset = {
            "samples":
            self.samples
        }

        json_path = os.path.join(
            self.audio_out,
            "dataset_kora_2025.json"
        )

        with open(
            json_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                dataset,
                f,
                indent=2
            )

        print(f"{self.counter} Archivos guardados en dataset")
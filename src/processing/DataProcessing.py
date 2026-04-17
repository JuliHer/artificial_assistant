import yaml
from pathlib import Path
import numpy as np
import tqdm
import os
import librosa
import random
class Process:
    def __init__(self):
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        with open(str(project_root / "src" / "configs" / "config.yaml"), 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        return

    def Bio_encoder_synchronized(self, text, entities, model):
        max_seq = self.config["ner_settings"]["max_words"]
        
        encoding = model.tokenizer(
            text, 
            return_offsets_mapping=True, 
            truncation=True, 
            max_length=max_seq,
            padding='max_length'
        )
        
        offsets = encoding['offset_mapping']
        labels = np.zeros(max_seq, dtype=int)
        
        # 2. Recorremos cada TOKEN (no cada palabra)
        for i, (start_token, end_token) in enumerate(offsets):
            # Ignorar tokens de padding o especiales que no mapean a texto (como BOS/EOS)
            if start_token == 0 and end_token == 0:
                continue
                
            for ent in entities:
                # Verificamos si el token actual está dentro del rango de la entidad
                if start_token < ent['end'] and end_token > ent['start']:
                    type_id = ent['type']
                    
                    # Lógica BIO: 
                    # Si el token empieza justo donde empieza la entidad -> B
                    # Si el token empieza después -> I
                    if start_token <= ent['start'] < end_token:
                        labels[i] = (type_id * 2) + 1
                    else:
                        labels[i] = (type_id * 2) + 2
                    break
                    
        return labels
    
    
    
    def embeddingBatch(self, model, texts):
        max_seq = self.config["ner_settings"]["max_words"]
        dim = self.config["ner_settings"]["embedding_dim"]

        final_embeddings = np.zeros((len(texts), max_seq, dim), dtype=np.float32)

        batch_size = 64

        all_token_embs = model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=True, 
            output_value='token_embeddings'
        )
        for i, emb in enumerate(tqdm.tqdm(all_token_embs, desc="Procesando NER Embeddings")):
            num_tokens = min(len(emb), max_seq)
            final_embeddings[i, :num_tokens, :] = emb[:num_tokens]

        return final_embeddings

    def dinamic_processing_samples(self, dir, sample_list, mode, seed, rate=16000,frame=32, hop=20, buffer_frames=64):
        random.seed(seed)
        hop_s = hop / 1000
        frame_s = frame /1000

        X, y = [], []

        padding = 300
        padding_s = padding/1000
        padding_len = int(padding_s/hop_s)

        buffer_duration = frame_s + (buffer_frames - 1) * hop_s
        buffer_len = int(buffer_duration/hop_s)
        buffers = []
        for sample in tqdm.tqdm(sample_list, f"Procesando audios {str(mode)}", unit="sample"):

            for file in sample["files"]:
                if not file["file"].endswith(".wav"):
                    continue
                audio_path = os.path.join(dir, file["file"])

                if not os.path.exists(audio_path):
                    print(f"Archivo no encontrado: {audio_path}")
                    continue

                signal, sr = librosa.load(audio_path, sr=rate)
                mel_full = self.mel_process(signal, int(hop_s*rate), rate)
                audio_len = len(signal)/rate

                if mode == "train":
                    wakes = []
                    for wake_interval in file["wake_intervals"]:
                        wake_start = int(wake_interval["start"]/hop_s)
                        wake_end = int(wake_interval["end"]/hop_s)
                        wake_len = int(wake_end - wake_start)
                        
                        wakes.append([wake_start, wake_end])
                        for i in range(6):
                            target_pos = 0
                            if i > 2:
                                target_pos = random.randint(0, buffer_len - wake_len) if wake_len < buffer_len else 0
                            else:
                                target_pos = random.randint(int(-wake_len*0.05), int(wake_len*0.05)) 


                            start = wake_start - target_pos
                            end = int(start + buffer_len)

                            if end > mel_full.shape[1] or start < 0:
                                continue
                            buffer = mel_full[:, start:end]
                            if buffer.shape[1] != buffer_len:
                                continue
                            buffers.append(buffer)

                            y.append(1)

                    steps = max(0, int((audio_len - buffer_duration) / padding_s)+1)

                    for pad in range(steps):
                        start = int(pad*padding_len)
                        end = int(start+buffer_len)

                        is_overlap = False
                        for wake in wakes:
                            if end > wake[0] and start < wake[1]:
                                is_overlap = True
                                break
                        
                        if is_overlap: continue

                        if end > mel_full.shape[1]:
                            break
                        buffer = mel_full[:, start: end]
                        if buffer.shape[1] != buffer_len:
                            continue
                        buffers.append(buffer)
                        y.append(0)
                        
                else: 
                    steps = max(0, int((audio_len - buffer_duration) / padding_s)+1)

                    for pad in range(steps):
                        start = int(pad*padding_len)
                        end = int(start+buffer_len)
                        if end > mel_full.shape[1]:
                            break
                        buffer = mel_full[:, start: end]
                        if buffer.shape[1] != buffer_len:
                            continue
                        buffers.append(buffer)
                        label = 0
                        for wake_interval in file["wake_intervals"]:
                            overlap = max(
                                0,
                                min(end*hop_s, wake_interval["end"]) -
                                max(start*hop_s, wake_interval["start"])
                            )
                            duration_wake = wake_interval["end"] - wake_interval["start"] 
                            if overlap >= 0.8 * duration_wake:
                                label = 1
                                break


                        y.append(label)
        
        X = np.array(buffers)
        y = np.array(y)

        return X, y
                    

    def mel_process(self, frame, hop, sr):
        mel_full = librosa.feature.melspectrogram(
            y=frame,
            sr=sr,
            n_fft=512,
            hop_length=hop,
            n_mels=40
        )

        mel_full = librosa.power_to_db(mel_full, ref=np.max)
      
        mel_full = (mel_full - np.mean(mel_full)) / (np.std(mel_full) + 1e-6)

        return mel_full
                    
    
    def getUniqueSortedMap(self, y_raw):
        unique = sorted(list(set(y_raw)))

        item_to_idx = {item: i for i, item in enumerate(unique)}
        idx_to_item = {i: item for i, item in enumerate(unique)}

        y_train = np.array([item_to_idx[item] for item in y_raw])

        return y_train, idx_to_item
    

    def indexToItem(self, idx, idx_to_item):
        return idx_to_item[idx]

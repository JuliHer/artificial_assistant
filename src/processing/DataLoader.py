from src.processing.DataProcessing import Process
from sentence_transformers import SentenceTransformer
from pathlib import Path
import os
import numpy as np
import json
import random

class Loader:
    def __init__(self):
        current_file = Path(__file__).resolve()
        self.project_root = current_file.parent.parent.parent
        self.ps = Process()
        self.action_map = []
        self.model = SentenceTransformer(str(self.project_root / "models" / "models--google--embeddinggemma-300m" / "snapshots" / "57c266a740f537b4dc058e1b0cda161fd15afa75"))


    def get_data_intentions(self):
        data = self.load_data()
        cache_data = self.load_cache("intention", data["_meta"]["version"])
        
        if cache_data is not False:
            X, y = cache_data
            return X, y
        else:
            X = []
            y = []

            intentions = []

            intentions.extend(data["intents"])

            sentences = []
            actions = []

            for item in intentions:
                for example in item["examples"]:
                    action = example["action"]
                    expression = example["expression"]

                    actions.append(action)
                    sentences.append(expression)

                    if action not in self.action_map:
                        self.action_map.append(action)

            
            print(f"Generando embeddings para {len(sentences)} frases...")
            X = self.model.encode(sentences, show_progress_bar=True)
            y = np.array(actions)
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            y, _ = self.ps.getUniqueSortedMap(y)

            self.save_cache("intention", data["_meta"]["version"], X, y)
            return X, y

        



    def get_data_wake_word(self, seed, rate=16000, frame=32, hop=20, buffer_frames=64):

        dir = self.project_root / "data" / "audio" / "processed" / "dataset" / "audio"

        if not os.path.exists(dir):
            print("El directorio de audio no existe")

        with open(dir / "dataset_kora_2025.json", "r", encoding="utf-8") as file:
            data = json.load(file)
        samples = data["samples"]
        random.seed(seed)
        random.shuffle(samples)
        
        n = len(samples)

        train_end = int(n*0.8)

        train_samples = samples[:train_end]
        test_samples = samples[train_end:]

        print(f"{len(train_samples)} train samples cargados")
        print(f"{len(test_samples)} test samples cargados")

        X_train, Y_train = self.ps.dinamic_processing_samples(dir, train_samples, "train", seed, rate, frame, hop, buffer_frames)
        X_test, Y_test = self.ps.dinamic_processing_samples(dir, test_samples, "test", seed, rate, frame, hop, buffer_frames)
        print(f"{len(X_train)} | {len(X_test)}")
        valores, conteos = np.unique(Y_train, return_counts=True)
        dict_conteos = dict(zip(valores, conteos))

        limit = 0.6

        X_train = X_train[:int(len(X_train)*limit)]
        Y_train = Y_train[:int(len(Y_train)*limit)]
        X_test = X_test[:int(len(X_test)*limit)]
        Y_test = Y_test[:int(len(Y_test)*limit)]

        negativos = dict_conteos.get(0, 0)
        positivos = dict_conteos.get(1, 0)

        print(f"Negativos (y=0): {negativos}")
        print(f"Positivos (y=1): {positivos}")
        indices = np.random.permutation(len(X_train))


        X_train = X_train[indices]
        Y_train = Y_train[indices]

        X_train = np.transpose(X_train, (0, 2, 1))
        X_test = np.transpose(X_test, (0, 2, 1))

        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]
        return X_train, X_test, Y_train, Y_test

        

    def load_data(self):
        dir = self.project_root / "data" / "processed"
        if not os.path.exists(dir):
            print("El directorio no existe.")
            return None
        
        with open(dir / "dataset_2025.json", "r", encoding="utf-8") as file:
            data = json.load(file)

        
        return data

    def get_data_ner(self):
        
        data = self.load_data()

        cache_data = self.load_cache("ner", data["_meta"]["version"])
        
        if cache_data is not False:
            X, y = cache_data
            return X, y
        else:
            X = []
            y = []

            intentions = []

            intentions.extend(data["intents"])


            all_expressions = []
            all_entities = []
            for item in intentions:
                for example in item["examples"]:
                    all_expressions.append(example["expression"])
                    all_entities.append(example.get("entities", []))


            y = np.array([self.ps.Bio_encoder_synchronized(exp, ent, self.model) for exp, ent in zip(all_expressions, all_entities)] )
            
            X = self.ps.embeddingBatch(self.model, all_expressions)
            indices = np.arange(X.shape[0])
            indices = np.random.permutation(len(X))

            # Barajamos X e y usando los índices de forma eficiente
            X = X[indices]
            y = y[indices]
            
            self.save_cache("ner", data["_meta"]["version"], X, y)
            return X, y

    def save_cache(self, model, version, X, y):
        cache_dir = self.project_root / "data" / "cache"
        x_cache_path = cache_dir / f"{model}_x_{version}.npy"
        y_cache_path = cache_dir / f"{model}_y_{version}.npy"
        print("Guardando embeddings en caché para futuras sesiones...")
        np.save(str(x_cache_path), X)
        np.save(str(y_cache_path), y)
        print("Caché guardado exitosamente.")

    def load_cache(self, model, version):
            cache_dir = self.project_root / "data" / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)

            x_cache_path = cache_dir / f"{model}_x_{version}.npy"
            y_cache_path = cache_dir / f"{model}_y_{version}.npy"

            if x_cache_path.exists() and y_cache_path.exists():
                print("--- Cargando embeddings desde el caché (.npy) ---")
                X = np.load(str(x_cache_path))
                y = np.load(str(y_cache_path))
                print(f"Carga completada: {X.shape[0]} muestras cargadas instantáneamente.")
                return X, y
            else:
                return False
        
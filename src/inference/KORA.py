import tensorflow as tf
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from pathlib import Path
from src.processing.DataProcessing import Process
import yaml 
from src.core.context import ActionContext
import src.utils.helper as helper

class KORA:

    id2ner = {
        0: "O", 
        1: "B-location",
        2: "I-location",
        3: "B-location_name",
        4: "I-location_name",
        5: "B-play",
        6: "I-play",
        7: "B-value",
        8: "I-value",
        9: "B-contact",
        10: "I-contact",
        11: "B-content",
        12: "I-content",
        13: "B-position",
        14: "I-position",
        15: "B-mode",
        16: "I-mode",
        17: "B-vehicle_port",
        18: "I-vehicle_port"
    }

    def __init__(self):
        current_file = Path(__file__).resolve()
        self.project_root = current_file.parent.parent.parent
        self.model = SentenceTransformer(str(self.project_root / "models" / "models--google--embeddinggemma-300m" / "snapshots" / "57c266a740f537b4dc058e1b0cda161fd15afa75"))
        
        self.load()
        pass

    def load(self):
        with open(str(self.project_root / "src" / "configs" / "config.yaml"), 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        ner_version = self.config["ner_settings"]["version"]
        intention_version = self.config["intention_settings"]["version"]
        

        ner_model_path = str(self.project_root / "models" / "ner" / f"KORA_{ner_version}.keras")
        intention_model_path = str(self.project_root / "models" / "intent_classifier" / f"KORA_{intention_version}.keras")
        if os.path.exists(intention_model_path):
            self.intention_model = tf.keras.models.load_model(intention_model_path)
        if os.path.exists(ner_model_path):
            self.ner_model = tf.keras.models.load_model(ner_model_path)

    def predict(self, text, as_context = True):
        max_seq = self.config["ner_settings"]["max_words"]
        embedding_dim = self.config["ner_settings"]["embedding_dim"]
        


        sentence_emb = self.model.encode([text])
        tokens_emb = self.model.encode(text, output_value='token_embeddings')


        word_emb_input = np.zeros((1, max_seq, embedding_dim))
        num_tokens = min(len(tokens_emb), max_seq)
        word_emb_input[0, :num_tokens, :] = tokens_emb[:num_tokens]

        intent_output = self.intention_model.predict(sentence_emb, verbose=0)
        intent_probs = intent_output[0]
        intent_idx = np.argmax(intent_probs)
        intent_confidence = float(intent_probs[intent_idx])
        intent = helper.get_intent_name(intent_idx)
        
    
        ner_output = self.ner_model.predict(word_emb_input, verbose=0)
        token_predictions = np.argmax(ner_output[0], axis=-1) # (max_seq,)


        encoding = self.model.tokenizer(text, 
                                        return_offsets_mapping=True,
                                        truncation=True, 
                                        max_length=max_seq)
        
        offsets = encoding['offset_mapping'][:num_tokens]
     
        
        
        entities = {}
        current_entity = None
        current_text = []
        current_scores = []
        for i, (start, end) in enumerate(offsets):
            if start == 0 and end == 0: continue # Ignorar BOS/EOS/Padding
            
            if i >= len(token_predictions): 
                break
            
            probs = ner_output[0][i]
            label_id = token_predictions[i]
            label = self.id2ner[label_id]

            score = float(probs[label_id])
            if label == "O":
                if current_entity:
                    entities[current_entity] = {
                        "value": " ".join(current_text),
                        "confidence": sum(current_scores) / len(current_scores),
                    }
                    current_entity, current_text, current_scores = None, [], []
                continue
            prefix, ent_type = label.split("-", 1)
            token_text = text[start:end]

            if prefix == "B":
                if current_entity:
                    entities[current_entity] = {
                        "value": "".join(current_text),
                        "confidence": sum(current_scores) / len(current_scores),
                    }

                current_entity = ent_type.lower()
                current_text = [token_text]
                current_scores = [score]
            elif prefix == "I" and current_entity == ent_type.lower():
                current_text.append(token_text)
                current_scores.append(score)
        
        if current_entity:
            entities[current_entity] = {
                "value": " ".join(current_text),
                "confidence": sum(current_scores) / len(current_scores),
                
            }

        raw_output = {
            "intent": {
                "name": intent,
                "confidence": intent_confidence
            },
            "entities": entities
        }

        if not as_context:
            return raw_output

        context = ActionContext(
                        intent=intent,
                        intent_confidence=intent_confidence,
                        entities=entities
                    )
        return context
    


from src.core.context import ActionContext
from src.configs.intent_map import INTENT_MAP
import numpy as np

def get_intent_name(code):
    return INTENT_MAP.get(code, "unknown_intent")

def get_entity(context:ActionContext, name, min_conf=0.5):
    ent = context.entities.get(name)
    if ent and ent["confidence"] >= min_conf:
        return ent["value"]
    return None

def get_weights(y):
    unique_indices, counts = np.unique(y, return_counts=True)
    
    max_data_len = np.max(counts)
    calculated_weights = np.sqrt(max_data_len / counts)
    
    weights = {int(idx): float(w) for idx, w in zip(unique_indices, calculated_weights)}
    
    return weights
    
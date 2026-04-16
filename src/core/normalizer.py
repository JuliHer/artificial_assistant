import numpy as np

def normalize(context, skill, model):
    intent = skill.intents.get(context.intent)

    if not intent or "entity_schema" not in intent:
        return context.entities
 
    translator = {}
    for ent_type, ent in context.entities.items():
        raw = ent["value"]

        

        schema = intent["entity_schema"].get(ent_type)

        if not schema:
            continue
        if schema.get("type") == "free":
            translator[raw] = raw
            continue

        translator[raw] = _normalize(
            model=model,
            ref=raw,
            values=schema["values"]
        )

    return translator


def _normalize(model, ref, values):
    if not values:
        return ref
    # Convertimos todo a texto (muy importante)
    ref = str(ref)
    values = [str(v) for v in values]

    # Embeddings
    ref_emb = model.encode(ref)              # (D,)
    values_emb = model.encode(values)        # (N, D)

    ref_norm = np.linalg.norm(ref_emb)

    if ref_norm == 0:
        return values[0]
    
    # Normalización L2 (Gemma no siempre viene normalizado)
    ref_emb = ref_emb / np.linalg.norm(ref_emb)
    values_emb = values_emb / np.linalg.norm(values_emb, axis=1, keepdims=True)

    # Similitud coseno
    similarities = np.dot(values_emb, ref_emb)   # (N,)

    # Índice del más cercano
    best_idx = int(np.argmax(similarities))

    return values[best_idx]


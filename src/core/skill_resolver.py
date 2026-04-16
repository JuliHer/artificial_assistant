def resolve_skill(context, skills):
    """
    Devuelve la mejor skill candidata o None
    """
    candidates = []

    for skill in skills:
        if context.intent not in skill.intents:
            continue

        score = _score_skill(skill, context)
        candidates.append((score, skill))

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (x[0], x[1].priority),
        reverse=True
    )

    return candidates[0][1]

def _score_skill(skill, context):
    score = 0

    # Base por intent match
    score += 10

    intent_cfg = skill.intents.get(context.intent, {})

    # Bonus por entidades opcionales presentes
    for entity in intent_cfg.get("optional_entities", []):
        if entity in context.entities:
            score += 2

    # Bonus por confianza
    score += int(context.intent_confidence * 10)

    return score
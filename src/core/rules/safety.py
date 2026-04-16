from src.core.policy_result import PolicyResult

def check(context, skill):
    intent_confidence = context.intent_confidence

    if skill.type == "hardware" and intent_confidence < 0.8:
        return PolicyResult(
            decision="ASK",
            message="Esta acción controla un dispositivo. ¿Confirmas?"
        )

    return PolicyResult("ALLOW")
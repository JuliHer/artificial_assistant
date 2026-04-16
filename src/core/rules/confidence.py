from src.core.policy_result import PolicyResult

MIN_CONF = 0.6

def check(context, skill):
    intent_confidence = context.intent_confidence

    if intent_confidence < MIN_CONF:
        return PolicyResult(
            decision="DENY",
            message="¿Puedes repetir o aclarar lo que quieres hacer?"
        )

    return PolicyResult("ALLOW")
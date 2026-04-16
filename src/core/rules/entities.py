from src.core.policy_result import PolicyResult

def check(context, skill):
    intent = context.intent
    intent_cfg = skill.intents.get(intent, {})

    required = intent_cfg.get("required_entities", [])
    missing = [e for e in required if e not in context.entities]

    if missing:
        return PolicyResult(
            decision="ASK",
            message=f"Falta {', '.join(missing)}",
            missing=missing
        )

    return PolicyResult("ALLOW")
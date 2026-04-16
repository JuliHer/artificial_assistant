from src.core.policy_result import PolicyResult

def check(context, skill):
    intent = context.intent

    intent_cfg = skill.intents.get(intent, {})

    permissions = intent_cfg.get("permissions")

    if not permissions:
        return None  

    user_role = context.session.get("role", "user")

    if user_role not in permissions:
        return PolicyResult(
            decision="DENY",
            message="No tienes permisos para esta acción"
        )

    return None

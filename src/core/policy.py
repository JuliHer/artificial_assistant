from src.core.rules import (
    confidence,
    entities,
    permissions,
    safety
)
from src.core.policy_result import PolicyResult

RULES = [
    confidence.check,
    entities.check,
    permissions.check,
    safety.check
]

def evaluate(context, skill):
    for rule in RULES:
        result = rule(context, skill)

        if result is None:
            continue

        if result.decision != "ALLOW":
            return result

    return PolicyResult("ALLOW")


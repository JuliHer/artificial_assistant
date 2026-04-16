
from src.inference.KORA import KORA

import src.core.skill_loader as loader
import src.core.skill_resolver as resolver
import src.core.policy as policy
import src.core.dispatcher as dispatcher

brain = KORA()

def test_play_music_flow():
    entrada = "Pon musica de"
    print("Usuario: ",entrada)
    context = brain.predict(entrada)

    skills = loader.load_skills()
    skill = resolver.resolve_skill(context, skills)
    
    context.debug.log("SKILL_RESOLVER", {
                        "candidates": [s.name for s in skills],
                        "selected": skill.name if skill else None
                    })
    
    
    decision = policy.evaluate(context, skill)
    context.debug.log("POLICY", {
                        "decision": decision.decision,
                        "reason": decision.message
                    })
    
    assert decision.decision == "ALLOW"

    result = dispatcher.dispatch(context, skill)

    context.debug.log("DISPATCH", {
                        "skill": skill.name,
                        "type": skill.type
                    })
    
    context.debug.dump()
    print(result)

test_play_music_flow()
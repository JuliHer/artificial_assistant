from src.logging.error_logger import log_error
from src.logging.hardware_logger import log_hardware
from src.logging.software_logger import log_software
from src.core.action_result import ActionResult, ActionStatus

def dispatch(context, skill):
    if not skill or not hasattr(skill, "handler"):
        return ActionResult(
            status=ActionStatus.FAILED,
            message="Skill Invalida"
        )

    try:
        result = skill.handler.execute(context)

        return result
    except Exception as e:
        log_error(e, context)
        return ActionResult(
            status=ActionStatus.FAILED,
            message="Ocurrio un error al ejecutar la accion"
        )
    
def resume(context, skill, pending_action):
    if context.intent == "cancelaction":
        return ActionResult(
            status=ActionStatus.FAILED,
            message="Acción cancelada"
        )
    try:
        if pending_action["type"] == ActionStatus.CONFIRM_REQUIRED:
            if context.intent == "confirmaction":
                return skill.execute(pending_action["data"])
            elif context.intent == "denyaction":
                return ActionResult(
                    status=ActionStatus.FAILED,
                    message="Acción cancelada"
                )
            else:
                return ActionResult(
                    status=ActionStatus.CONFIRM_REQUIRED,
                    message="¿Confirmas la acción?",
                    data=pending_action["data"]
                )
        
        if pending_action["type"] in [ActionStatus.CONFIRM_REQUIRED, ActionStatus.NEEDS_INPUT, "ASK"]:
            new_data = merge_entities(
                pending_action["data"],
                context.entities
            )

            missing = check_missing(context, skill, new_data)

            if missing:
                return ActionResult(
                    status=ActionStatus.NEEDS_INPUT,
                    message=skill.ask_for(missing),
                    data=new_data,
                    missing=missing
                )

            if skill.confirmation:
                return ActionResult(
                    status=ActionStatus.CONFIRM_REQUIRED,
                    message=skill.confirm_message(new_data),
                    data=new_data
                )
    
        result = skill.handler.execute(context)

        return result
    except Exception as e:
        log_error(e, context)
        return ActionResult(
            status=ActionStatus.FAILED,
            message="Ocurrio un error al ejecutar la accion"
        )
    
def merge_entities(old: dict, new: dict) -> dict:
    merged = old.copy()
    merged.update(new)
    return merged

def check_missing(context, skill, data):
    intent = context.intent
    intent_cfg = skill.intents.get(intent, {})

    required = intent_cfg.get("required_entities", [])
    missing = [e for e in required if e not in data]

    return missing

  
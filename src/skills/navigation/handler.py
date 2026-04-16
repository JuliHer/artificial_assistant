from .routes import NAVIGATION_ACTIONS
from src.core.action_result import ActionResult, ActionStatus

class NavigationHandler:
    def __init__(self):
        pass

    def execute(self, context):
        action = NAVIGATION_ACTIONS.get(context.intent)

        if not action:
            return ActionResult(
            status=ActionStatus.FAILED,
            message="Accion de navegacion no soportada"
        )
        
        return ActionResult(
            status=ActionStatus.SUCCESS,
            message="Navigation Realized"
        )
        
    def _error(self, details, context):
        from src.logging.error_logger import log_error
    
        log_error(details, context)
        return ActionResult(
            status=ActionStatus.FAILED,
            message=details
        )
from .routes import MEDIA_ACTIONS
from src.core.action_result import ActionResult, ActionStatus

class MediaControlHandler:
    def __init__(self):
        pass

    def execute(self, context):
        action = MEDIA_ACTIONS.get(context.intent)

        if not action:
            return self._error("Accion de audio no soportada", context)

        # Decide backend
        backend = self._select_backend(context)

        if backend == "hardware":
            return self._exec_hardware(action, context)

        return self._exec_software(action, context)

    def _select_backend(self, context):
        source = context.entities.get("source")

        if source in ["radio", "bluetooth", "aux"]:
            return "hardware"

        return "software"
    
    def _exec_hardware(self, action, context):
        from src.logging.hardware_logger import log_hardware

        log_hardware(
            action=f"MEDIA_{action.upper()}",
            device=context.entities.get("source", "audio_hw"),
            user=context.user
        )

        #self.hardware.execute(action, context.entities)

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message="Control de audio aplicado"
        )
    
    def _error(self, details, context):
        from src.logging.error_logger import log_error

        log_error(details, context)
        return ActionResult(
            status=ActionStatus.FAILED,
            message=details
        )

    def _exec_software(self, action, context):
        #result = self.software.execute(action, context.entities)

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message="Hecho"
        )
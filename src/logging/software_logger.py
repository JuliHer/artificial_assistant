from .base_logger import BaseLogger

_software_logger = BaseLogger("software.log")

def log_software(action, detail=None, context=None):
    record = {
        "type": "software",
        "action": action,
        "detail": detail,
        "intent": getattr(context, "intent", None),
        "user": getattr(context, "user", None)
    }

    _software_logger._write(record)
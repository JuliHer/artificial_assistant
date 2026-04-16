from .base_logger import BaseLogger
import traceback

_error_logger = BaseLogger("errors.log")


def log_error(error, context=None):
    record = {
        "type": "error",
        "error": str(error),
        "traceback": traceback.format_exc(),
        "intent": getattr(context, "intent", None),
        "user": getattr(context, "user", None)
    }

    _error_logger._write(record)
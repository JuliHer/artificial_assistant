from .base_logger import BaseLogger

_hardware_logger = BaseLogger("hardware.log")


def log_hardware(action, device, user, entities=None):
    record = {
        "type": "hardware",
        "action": action,
        "device": device,
        "user": user,
        "entities": entities
    }

    _hardware_logger._write(record)
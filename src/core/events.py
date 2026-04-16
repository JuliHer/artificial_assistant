from dataclasses import dataclass
from enum import Enum, auto

class EventType(Enum):
    WAKE_WORD = auto()
    COMMAND_AUDIO = auto()

class Event:
    type: EventType
    payload: dict
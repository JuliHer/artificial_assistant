from enum import Enum, auto

class AudioState(Enum):
    INIT = auto()
    CALIBRATING = auto()
    IDLE = auto()
    LISTENING = auto()
    TRIGGERED = auto()
    RECORDING = auto()
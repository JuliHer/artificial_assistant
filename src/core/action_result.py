from dataclasses import dataclass
from typing import Any, Dict, Optional

class ActionStatus:
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CONFIRM_REQUIRED = "CONFIRM_REQUIRED"
    NEEDS_INPUT = "NEEDS_INPUT"
    DEFERRED = "DEFERRED"

class ActionResult:
    def __init__(self, status:str,message: str = None, data: Dict[str, Any] = None, confidence: float = None, action_id: str = None, requires_confirmation: bool = False):
        self.status = status
        self.message = message
        self.data = data
        self.confidence = confidence
        self.action_id = action_id
        self.requires_confirmation = requires_confirmation
        pass

    
    
    
    
    
    
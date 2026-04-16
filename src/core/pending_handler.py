import time 
from typing import Any, Dict
class PendingHandler:

    def create_pending(self, context, skill, type, message:str, data:Dict[str, Any] = None, missing = None):
        context.session["pending_action"] = {
            "id": f"pending_{int(time.time()*1000)}",
            "skill": skill,
            "message": message,
            "intent": context.intent,
            "type": type,              # NEEDS_INPUT | CONFIRM_REQUIRED
            "missing": missing or [],
            "data": data or {},
            "timestamp": time.time(),
            "timeout": 15
        }
    
    def is_pending(self, context):
        return "skill" in context.session.get("pending_action", [])

    def is_timed_out(self, context):
        ts = context.session.get("timestamp")
        timeout = context.session.get("pending_action",[]).get("timeout", 15)

        if not ts:
            return False

        return (time.time() - ts) > timeout

    def clear(self, context):
        if "pending_action" in context.session:
            for k in ["skill", "intent", "type", "message", "missing", "data", "timestamp", "timeout"]:
                context.session["pending_action"].pop(k, None)
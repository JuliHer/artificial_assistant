import time
from src.core.debug import DebugContext

class ActionContext:
    def __init__(self, intent, intent_confidence, entities, raw_text = None, session = None, user = "default"):

        self.intent = intent
        self.intent_confidence = intent_confidence
        self.entities = entities  
        self.session = session or {}
        self.timestamp = time.time()
        self.raw_text = raw_text

        # Conversacional / extensible
        self.user = user
        self.debug = DebugContext()
        pass
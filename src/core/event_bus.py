

class EventBus:

    def __init__(self):
        
        self._subscribers = {}

    def subscribe(self, eventType, callback):
        self._subscribers.setdefault(eventType, []).append(callback)

    def emit(self, event):
        for cb in self._subscribers.get(event.type, []):
            cb(event)


class Skill:
    def __init__(
        self,
        name,
        intents,
        skill_type,
        handler,
        confirmation=False,
        priority=0
    ):
        self.name = name
        self.intents = intents
        self.type = skill_type
        self.handler = handler
        self.confirmation = confirmation
        self.priority = priority
class PolicyResult:
    def __init__(self, decision, message=None, missing=None):
        self.decision = decision  
        self.message = message
        self.missing = missing
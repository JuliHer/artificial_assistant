class DebugContext:
    def __init__(self):
        self.steps = []

    def log(self, step, data):
        self.steps.append({
            "step": step,
            "data": data
        })

    def dump(self):
        for s in self.steps:
            print(f"[DEBUG::{s['step']}]")
            print(s["data"])
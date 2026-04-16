class MediaService:
    def execute(self, action, entities):
        if action == "set_volume":
            return f"Volumen ajustado a {entities.get('volume')}%"

        return f"Acción {action} ejecutada"
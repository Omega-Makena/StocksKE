class BaseNotifier:
    def send(self, message: str) -> bool:
        raise NotImplementedError("Subclasses must implement send method")

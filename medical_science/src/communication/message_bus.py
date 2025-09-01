from typing import Callable, Dict, List


class MessageBus:
    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, topic: str, handler: Callable) -> None:
        self._subscribers.setdefault(topic, []).append(handler)

    def publish(self, topic: str, message) -> None:
        for handler in self._subscribers.get(topic, []):
            handler(message) 
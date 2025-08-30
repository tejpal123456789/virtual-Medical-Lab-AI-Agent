from .message_bus import MessageBus


class AgentCommunicator:
    def __init__(self, bus: MessageBus | None = None) -> None:
        self.bus = bus or MessageBus()

    def send(self, agent: str, payload: dict) -> None:
        self.bus.publish(agent, payload) 
from dataclasses import dataclass


@dataclass
class MCPMessage:
    sender: str
    recipient: str
    content: str 
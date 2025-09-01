from abc import ABC, abstractmethod
from typing import Any


class ChatLLM(ABC):
    @abstractmethod
    def invoke(self, prompt: str) -> Any:
        raise NotImplementedError


class EmbeddingModel(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError 
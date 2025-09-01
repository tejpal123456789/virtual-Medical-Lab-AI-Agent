import os
import json
from typing import List, Any
import requests
from langchain_core.embeddings import Embeddings


class StellaEmbeddings(Embeddings):
    def __init__(self, api_key: str | None = None, model_url: str | None = None, vector_dim: int = 1024):
        self.api_key = api_key or os.getenv("stella_api_key") or os.getenv("SIMPLISMART_STELLA_API_KEY")
        self.model_url = model_url or os.getenv("stella_model_url") or os.getenv("SIMPLISMART_STELLA_URL")
        self.vector_dim = vector_dim
        if not self.api_key or not self.model_url:
            raise ValueError("stella_api_key/SIMPLISMART_STELLA_API_KEY and stella_model_url/SIMPLISMART_STELLA_URL must be set for StellaEmbeddings")

    def _encode(self, texts: List[str], prefix: str) -> List[List[float]]:
        payload = json.dumps({
            "query": texts,
            "prefix": prefix,
            "vector_dim": self.vector_dim,
        })
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        response = requests.post(self.model_url, headers=headers, data=payload, timeout=60)
        response.raise_for_status()
        embeddings_data = response.json()["embedding"]
        if isinstance(embeddings_data, list) and embeddings_data and isinstance(embeddings_data[0], (int, float)):
            return [embeddings_data]
        return embeddings_data

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts, prefix="passage")

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text], prefix="query")[0] 
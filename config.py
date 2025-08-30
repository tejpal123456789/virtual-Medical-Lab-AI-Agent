"""
Configuration file for the Multi-Agent Medical Chatbot

This file contains all the configuration parameters for the project.

If you want to change the LLM and Embedding model:

you can do it by changing all 'llm' and 'embedding_model' variables present in multiple classes below.

Each llm definition has unique temperature value relevant to the specific class. 
"""
import os
import json
from typing import List, Any
import requests

import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI, ChatOpenAI
from langchain_core.embeddings import Embeddings

# Load environment variables from .env file
load_dotenv()

class EmbeddingConfig:
    def __init__(self):
        self.provider = os.getenv("EMBEDDING_PROVIDER", "azure_openai").lower()
        # Stella settings
        self.stella_api_key = os.getenv("SIMPLISMART_STELLA_API_KEY")
        self.stella_model_url = os.getenv("SIMPLISMART_STELLA_URL")
        self.stella_vector_dim = int(os.getenv("stella_vector_dim", "1024"))

class AgentDecisoinConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model = "gpt-4o-mini",
            temperature = 0.1,
            api_key = os.getenv("openai_api_key")
        )

class ConversationConfig:
    def __init__(self):
         self.llm = ChatOpenAI(
            model = "gpt-4o-mini",
            temperature = 0.1,
            api_key = os.getenv("openai_api_key")
        )

class WebSearchConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model = "gpt-4o-mini",
            temperature = 0.1,
            api_key = os.getenv("openai_api_key")
        )
        self.context_limit = 20     # include last 20 messsages (10 Q&A pairs) in history

class StellaEmbeddings(Embeddings):
    def __init__(self, api_key: str | None = None, model_url: str | None = None, vector_dim: int = 1024):
        self.api_key = api_key or os.getenv("SIMPLISMART_STELLA_API_KEY")
        self.model_url = model_url or os.getenv("SIMPLISMART_STELLA_URL")
        self.vector_dim = vector_dim  
        if not self.api_key or not self.model_url:      
            raise ValueError("SIMPLISMART_STELLA_API_KEY and SIMPLISMART_STELLA_URL must be set for StellaEmbeddings")

    def _encode(self, texts: List[str], prefix: str) -> List[List[float]]:
        try:
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
            
            response_data = response.json()
            if "embedding" not in response_data:
                raise ValueError(f"Response missing 'embedding' key. Response: {response_data}")
            
            embeddings_data = response_data["embedding"]
            
            # Ensure list[list[float]]
            if isinstance(embeddings_data, list) and embeddings_data and isinstance(embeddings_data[0], (int, float)):
                return [embeddings_data]
            return embeddings_data
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error calling Stella API: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing Stella API response: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        return self._encode(texts, prefix="passage")

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self._encode([text], prefix="query")[0]


class RAGConfig:
    def __init__(self):
        self.vector_db_type = "qdrant"
        self.distance_metric = "Cosine"
        self.use_local = True
        self.vector_local_path = "./data/qdrant_db"
        self.doc_local_path = "./data/docs_db"
        self.parsed_content_dir = "./data/parsed_docs"
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = "medical_assistance_rag_v1"
        self.chunk_size = 512
        self.chunk_overlap = 50

        # Embeddings provider selection
        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", "stella")
        if self.embedding_provider == "stella":
            # Get vector dimension from environment with proper fallback
            vector_dim = int(os.getenv("STELLA_VECTOR_DIM", "1024"))
            
            self.embedding_model = StellaEmbeddings(
                api_key=os.getenv("SIMPLISMART_STELLA_API_KEY"),
                model_url=os.getenv("SIMPLISMART_STELLA_URL"),
                vector_dim=vector_dim,
            )
            self.embedding_dim = vector_dim
        else: 
            raise ValueError("No valid embedding provider configured. Set EMBEDDING_PROVIDER=stella")

        # Validate OpenAI API key
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=openai_key
        )
        self.summarizer_model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=openai_key
        )
        self.chunker_model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=openai_key
        )
        self.response_generator_model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=openai_key
        )
        
        self.top_k = 5
        self.vector_search_type = 'similarity'
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        self.reranker_model = "cross-encoder/ms-marco-TinyBERT-L-6"
        self.reranker_top_k = 3
        self.max_context_length = 8192
        self.include_sources = True
        self.min_retrieval_confidence = 0.40
        self.context_limit = 20

class MedicalCVConfig:
    def __init__(self):
        self.brain_tumor_model_path = "./agents/image_analysis_agent/brain_tumor_agent/models/brain_tumor_segmentation.pth"
        self.chest_xray_model_path = "./agents/image_analysis_agent/chest_xray_agent/models/covid_chest_xray_model.pth"
        self.skin_lesion_model_path = "./agents/image_analysis_agent/skin_lesion_agent/models/checkpointN25_.pth.tar"
        self.skin_lesion_segmentation_output_path = "./uploads/skin_lesion_output/segmentation_plot.png"
        self.llm = ChatOpenAI(
            model = "gpt-4o-mini",
            temperature = 0.1,
            api_key = os.getenv("openai_api_key")
        )

class SpeechConfig:
    def __init__(self):
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")  # Replace with your actual key
        self.eleven_labs_voice_id = "21m00Tcm4TlvDq8ikWAM"    # Default voice ID (Rachel)

class ValidationConfig:
    def __init__(self):
        self.require_validation = {
            "CONVERSATION_AGENT": False,
            "RAG_AGENT": False,
            "WEB_SEARCH_AGENT": False,
            "BRAIN_TUMOR_AGENT": True,
            "CHEST_XRAY_AGENT": True,
            "SKIN_LESION_AGENT": True
        }
        self.validation_timeout = 300
        self.default_action = "reject"

class APIConfig:
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 8000
        self.debug = True
        self.rate_limit = 10
        self.max_image_upload_size = 5  # max upload size in MB

class MemoryConfig:
    def __init__(self):
        # mem0 configuration
        self.provider = "mem0"
        self.collection_name = "medical_assistant_longterm_memory"
        
        # Qdrant settings for mem0 vector store
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        
        # Neo4j settings for mem0 graph store (optional)
        self.neo4j_url = os.getenv("NEO4J_URL")  # e.g., "bolt://localhost:7687"
        self.neo4j_username = os.getenv("NEO4J_USERNAME")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        # LLM settings for memory processing
        self.llm_model = "gpt-4o-mini"
        self.embedding_model = "text-embedding-3-small"  # Supports dimensions parameter
        self.embedding_dims = 1536  # Dimensions for text-embedding-3-small
        
        # Memory management settings
        self.max_memories_per_query = 5
        self.memory_retention_days = 90  # Keep memories for 90 days
        self.enable_auto_cleanup = True
        self.backup_enabled = True
        self.backup_interval_hours = 24
        
        # Memory enhancement settings
        self.enhance_conversation_prompts = True
        self.enhance_rag_prompts = True
        self.enhance_research_prompts = True
        self.max_context_length = 1000  # Max chars from memory to include in prompts


class UIConfig:
    def __init__(self):
        self.theme = "light"
        # self.max_chat_history = 50
        self.enable_speech = True
        self.enable_image_upload = True

class Config:
    def __init__(self):
        self.embedding = EmbeddingConfig()
        self.agent_decision = AgentDecisoinConfig()
        self.conversation = ConversationConfig()
        self.rag = RAGConfig()
        self.medical_cv = MedicalCVConfig()
        self.web_search = WebSearchConfig()
        self.memory = MemoryConfig()  # Add memory configuration
        self.api = APIConfig()
        self.speech = SpeechConfig()
        self.validation = ValidationConfig()
        self.ui = UIConfig()
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.max_conversation_history = 20  # Include last 20 messsages (10 Q&A pairs) in history

# # Example usage
# config = Config()




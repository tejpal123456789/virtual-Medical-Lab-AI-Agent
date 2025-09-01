import os
import re
import json
import requests
import logging
from uuid import uuid4
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain.storage import InMemoryStore, LocalFileStore
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams, OptimizersConfigDiff
from dotenv import load_dotenv
load_dotenv()

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
        self.collection_name = "demo_rag"
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


class VectorStore:
    """
    Create vector store, ingest documents, retrieve relevant documents
    """
    def __init__(self, config: RAGConfig):
        self.logger = logging.getLogger(__name__)
        self.collection_name = config.collection_name
        self.embedding_dim = config.embedding_dim
        self.distance_metric = config.distance_metric
        self.embedding_model = config.embedding_model
        self.retrieval_top_k = config.top_k
        self.vector_search_type = config.vector_search_type
        self.vectorstore_local_path = config.vector_local_path
        self.docstore_local_path = config.doc_local_path

        # Ensure directories exist
        os.makedirs(self.vectorstore_local_path, exist_ok=True)
        os.makedirs(self.docstore_local_path, exist_ok=True)
        
        self.client = QdrantClient(path=self.vectorstore_local_path)

    def _does_collection_exist(self) -> bool:
        """Check if the collection already exists in Qdrant."""
        try:
            collection_info = self.client.get_collections()
            collection_names = [collection.name for collection in collection_info.collections]
            return self.collection_name in collection_names
        except Exception as e:
            self.logger.error(f"Error checking for collection existence: {e}")
            return False

    def _create_collection(self):
        """Create a new collection with dense and sparse vectors."""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={"dense": VectorParams(size=self.embedding_dim, distance=Distance.COSINE)},
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
                },
            )
            self.logger.info(f"Created new collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Error creating collection: {e}")
            raise e
            
    def load_vectorstore(self) -> Tuple[QdrantVectorStore, LocalFileStore]:
        """
        Load existing vectorstore and docstore for retrieval operations without ingesting new documents.
        
        Returns:
            Tuple containing (vectorstore, docstore)
        """
        # Check if collection exists
        if not self._does_collection_exist():
            self.logger.error(f"Collection {self.collection_name} does not exist. Please ingest documents first.")
            raise ValueError(f"Collection {self.collection_name} does not exist")
            
        # Setup sparse embeddings
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        
        # Initialize vector store
        qdrant_vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        
        # Document storage
        docstore = LocalFileStore(self.docstore_local_path)
        
        self.logger.info(f"Successfully loaded existing vectorstore and docstore")
        return qdrant_vectorstore, docstore

    def create_vectorstore(
            self,
            document_chunks: List[str],
            document_path: str,
        ) -> Tuple[QdrantVectorStore, LocalFileStore, List[str]]:
        """
        Create a vector store from document chunks or upsert documents to existing store.
        
        Args:
            document_chunks: List of document chunks
            document_path: Path to the original document
            
        Returns:
            Tuple containing (vectorstore, docstore, doc_ids)
        """
        
        if not document_chunks:
            raise ValueError("document_chunks cannot be empty")
        
        # Generate unique IDs for each chunk
        doc_ids = [str(uuid4()) for _ in range(len(document_chunks))]
        
        # Create langchain documents
        langchain_documents = []
        for id_idx, chunk in enumerate(document_chunks):
            if not chunk.strip():  # Skip empty chunks
                continue
                
            langchain_documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": os.path.basename(document_path),
                        "doc_id": doc_ids[id_idx],
                        "source_path": os.path.join("http://localhost:8000/", document_path)
                    }
                )
            )
        
        if not langchain_documents:
            raise ValueError("No valid document chunks to process")
        
        # Setup sparse embeddings
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        
        # Check if collection exists, create if it doesn't
        collection_exists = self._does_collection_exist()
        if not collection_exists:
            self._create_collection()
            self.logger.info(f"Created new collection: {self.collection_name}")
        else:
            self.logger.info(f"Collection {self.collection_name} already exists, will upsert documents")
        
        # Initialize vector store
        qdrant_vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        
        # Document storage for parent documents
        docstore = LocalFileStore(self.docstore_local_path)
        
        try:
            # Ingest documents into vector and doc stores
            qdrant_vectorstore.add_documents(documents=langchain_documents, ids=doc_ids[:len(langchain_documents)])
            
            # Encode string chunks to bytes before storing
            valid_chunks = [chunk for chunk in document_chunks if chunk.strip()]
            encoded_chunks = [chunk.encode('utf-8') for chunk in valid_chunks]
            docstore.mset(list(zip(doc_ids[:len(encoded_chunks)], encoded_chunks)))
            
            self.logger.info(f"Successfully ingested {len(langchain_documents)} documents")
            return qdrant_vectorstore, docstore, doc_ids[:len(langchain_documents)]
            
        except Exception as e:
            self.logger.error(f"Error during document ingestion: {e}")
            raise

    def retrieve_relevant_chunks(
            self,
            query: str,
            vectorstore: QdrantVectorStore,
            docstore: LocalFileStore,
        ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks based on a query.
        
        Args:
            query: User query
            vectorstore: Vector store containing embeddings
            docstore: Document store containing actual content
            
        Returns:
            List of dictionaries with content and score
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        try:
            # Use similarity_search_with_score to get documents and scores
            results = vectorstore.similarity_search_with_score(
                query=query,
                k=self.retrieval_top_k
            )
            
            retrieved_docs = []
            
            for chunk, score in results:
                try:
                    # Get full document from doc store as bytes and decode to string
                    doc_content_bytes = docstore.mget([chunk.metadata['doc_id']])[0]
                    if doc_content_bytes is None:
                        self.logger.warning(f"Document with ID {chunk.metadata['doc_id']} not found in docstore")
                        continue
                        
                    doc_content = doc_content_bytes.decode('utf-8')
                    
                    # Create document dict in the format expected by reranker
                    doc_dict = {
                        "id": chunk.metadata['doc_id'],
                        "content": doc_content,
                        "score": float(score),  # Ensure score is serializable
                        "source": chunk.metadata['source'],
                        "source_path": chunk.metadata['source_path'],
                    }
                    retrieved_docs.append(doc_dict)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing chunk {chunk.metadata.get('doc_id', 'unknown')}: {e}")
                    continue
            
            return retrieved_docs
            
        except Exception as e:
            self.logger.error(f"Error during retrieval: {e}")
            raise


# Debug version to show when methods are called
class DebugStellaEmbeddings(StellaEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        print(f"🔍 embed_documents called with {len(texts)} texts")
        print(f"📝 First text preview: {texts[0][:100]}..." if texts else "No texts")
        result = super().embed_documents(texts)
        print(f"✅ embed_documents returned {len(result)} embeddings")
        return result

    def embed_query(self, text: str) -> List[float]:
        print(f"🔍 embed_query called with text: {text[:100]}...")
        result = super().embed_query(text)
        print(f"✅ embed_query returned embedding of length {len(result)}")
        return result

    def _encode(self, texts: List[str], prefix: str) -> List[List[float]]:
        print(f"🚀 _encode called with prefix='{prefix}' and {len(texts)} texts")
        result = super()._encode(texts, prefix)
        print(f"✅ _encode returned {len(result)} embeddings")
        return result


# Usage example showing the call flow
def main():
    # Initialize configuration
    config = RAGConfig()
    
    # Replace with debug version to see method calls
    config.embedding_model = DebugStellaEmbeddings(
        api_key=os.getenv("SIMPLISMART_STELLA_API_KEY"),
        model_url=os.getenv("SIMPLISMART_STELLA_URL"),
        vector_dim=config.embedding_dim,
    )
    
    # Initialize vector store
    vector_store = VectorStore(config)
    
    # Example document chunks
    document_chunks = [
        "This is the first chunk of the document.",
        "This is the second chunk with more information.",
        "This is the third chunk containing additional details."
    ]
    
    document_path = "example_document.txt"
    
    try:
        print("=== CREATING VECTOR STORE ===")
        # Create vector store - this will call embed_documents()
        qdrant_vectorstore, docstore, doc_ids = vector_store.create_vectorstore(
            document_chunks=document_chunks,
            document_path=document_path
        )
        
        print("\n=== PERFORMING QUERY ===")
        # Test retrieval - this will call embed_query()
        query = "information about the document"
        results = vector_store.retrieve_relevant_chunks(
            query=query,
            vectorstore=qdrant_vectorstore,
            docstore=docstore
        )
        
        print(f"\nRetrieved {len(results)} relevant chunks")
        for i, result in enumerate(results):
            print(f"Chunk {i+1}: {result['content'][:100]}... (Score: {result['score']:.4f})")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
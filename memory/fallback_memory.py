"""
Fallback memory system for when mem0 configuration fails.

This provides basic memory functionality using simple in-memory storage
or file-based persistence as a backup when mem0 is not available.
"""

import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SimpleFallbackMemory:
    """
    Simple fallback memory system that stores memories in JSON files.
    """
    
    def __init__(self, storage_dir: str = "./memory/fallback_storage"):
        """Initialize the fallback memory system."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.memories = {}  # In-memory cache
        self._load_memories()
    
    def _get_user_file(self, user_id: str) -> Path:
        """Get the storage file path for a user."""
        safe_user_id = "".join(c for c in user_id if c.isalnum() or c in "._-")
        return self.storage_dir / f"user_{safe_user_id}.json"
    
    def _load_memories(self):
        """Load all user memories from disk."""
        try:
            for file_path in self.storage_dir.glob("user_*.json"):
                user_id = file_path.stem.replace("user_", "")
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.memories[user_id] = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load some memories: {e}")
    
    def _save_user_memories(self, user_id: str):
        """Save memories for a specific user to disk."""
        try:
            user_file = self._get_user_file(user_id)
            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(self.memories.get(user_id, []), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save memories for user {user_id}: {e}")
    
    def add(self, content: str, user_id: str, metadata: Dict[str, Any] = None) -> str:
        """Add a memory entry."""
        memory_id = str(uuid.uuid4())
        
        memory_entry = {
            "id": memory_id,
            "memory": content,
            "user_id": user_id,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "score": 1.0
        }
        
        if user_id not in self.memories:
            self.memories[user_id] = []
        
        self.memories[user_id].append(memory_entry)
        self._save_user_memories(user_id)
        
        logger.info(f"✅ Stored memory {memory_id} for user {user_id}")
        return memory_id
    
    def search(self, query: str, user_id: str, limit: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search memories for a user."""
        if user_id not in self.memories:
            return []
        
        user_memories = self.memories[user_id]
        
        # Simple text matching for search
        query_lower = query.lower()
        matching_memories = []
        
        for memory in user_memories:
            memory_text = memory.get("memory", "").lower()
            
            # Apply filters if provided
            if filters:
                filter_match = True
                for filter_key, filter_value in filters.items():
                    if filter_key in memory.get("metadata", {}):
                        if isinstance(filter_value, list):
                            if memory["metadata"][filter_key] not in filter_value:
                                filter_match = False
                                break
                        else:
                            if memory["metadata"][filter_key] != filter_value:
                                filter_match = False
                                break
                if not filter_match:
                    continue
            
            # Simple relevance scoring based on keyword matches
            score = 0.0
            query_words = query_lower.split()
            for word in query_words:
                if word in memory_text:
                    score += 1.0
            
            if score > 0:
                memory_copy = memory.copy()
                memory_copy["score"] = score / len(query_words)  # Normalize score
                matching_memories.append(memory_copy)
        
        # Sort by score and return top results
        matching_memories.sort(key=lambda x: x["score"], reverse=True)
        return matching_memories[:limit]
    
    def get_all(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all memories for a user."""
        return self.memories.get(user_id, [])
    
    def delete(self, memory_id: str) -> bool:
        """Delete a specific memory."""
        for user_id, user_memories in self.memories.items():
            for i, memory in enumerate(user_memories):
                if memory.get("id") == memory_id:
                    del user_memories[i]
                    self._save_user_memories(user_id)
                    logger.info(f"✅ Deleted memory {memory_id}")
                    return True
        
        logger.warning(f"Memory {memory_id} not found for deletion")
        return False
    
    def cleanup_old_memories(self, user_id: str, days_to_keep: int = 30) -> int:
        """Clean up old memories."""
        if user_id not in self.memories:
            return 0
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        user_memories = self.memories[user_id]
        
        original_count = len(user_memories)
        
        # Filter out old memories
        self.memories[user_id] = [
            memory for memory in user_memories
            if datetime.fromisoformat(memory.get("timestamp", datetime.now().isoformat())) >= cutoff_date
        ]
        
        deleted_count = original_count - len(self.memories[user_id])
        
        if deleted_count > 0:
            self._save_user_memories(user_id)
            logger.info(f"✅ Cleaned up {deleted_count} old memories for user {user_id}")
        
        return deleted_count


class FallbackMemoryManager:
    """
    Fallback memory manager that provides the same interface as LongTermMemoryManager
    but uses simple file-based storage when mem0 is not available.
    """
    
    def __init__(self, config):
        """Initialize the fallback memory manager."""
        self.config = config
        self.memory = SimpleFallbackMemory()
        logger.info("✅ Fallback memory system initialized")
    
    def generate_user_id(self, session_info: Dict[str, Any] = None) -> str:
        """Generate or retrieve user ID for session tracking."""
        if session_info and "user_id" in session_info:
            return session_info["user_id"]
        return "default_user"
    
    def store_conversation_context(self, 
                                 user_id: str,
                                 messages: List,
                                 agent_name: str,
                                 metadata: Dict[str, Any] = None) -> str:
        """Store conversation context."""
        try:
            content = f"Conversation with {agent_name}"
            return self.memory.add(content, user_id=user_id, metadata=metadata)
        except Exception as e:
            logger.error(f"Failed to store conversation context: {e}")
            return ""
    
    def store_medical_insight(self,
                            user_id: str, 
                            insight: str,
                            source_agent: str,
                            confidence: float = 1.0,
                            metadata: Dict[str, Any] = None) -> str:
        """Store medical insights."""
        try:
            return self.memory.add(
                insight, 
                user_id=user_id, 
                metadata={**(metadata or {}), "source_agent": source_agent, "confidence": confidence}
            )
        except Exception as e:
            logger.error(f"Failed to store medical insight: {e}")
            return ""
    
    def retrieve_relevant_memories(self,
                                 user_id: str,
                                 query: str,
                                 memory_types: List = None,
                                 limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant memories."""
        try:
            filters = {}
            if memory_types:
                filters["memory_type"] = [mt.value if hasattr(mt, 'value') else str(mt) for mt in memory_types]
            
            return self.memory.search(query, user_id=user_id, limit=limit, filters=filters)
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []
    
    def get_user_profile_summary(self, user_id: str) -> str:
        """Get user profile summary."""
        try:
            memories = self.memory.get_all(user_id)
            if not memories:
                return "No previous interactions found for this user."
            
            return f"User has {len(memories)} previous interactions."
        except Exception as e:
            logger.error(f"Failed to get user profile: {e}")
            return "Unable to retrieve user profile."
    
    def enhance_prompt_with_memory(self,
                                 user_id: str,
                                 current_query: str,
                                 agent_name: str,
                                 base_prompt: str) -> str:
        """Enhanced prompt with simple memory context."""
        try:
            relevant_memories = self.retrieve_relevant_memories(user_id, current_query, limit=2)
            
            if relevant_memories:
                memory_context = "\n**Previous Context:**\n"
                for i, memory in enumerate(relevant_memories, 1):
                    memory_context += f"{i}. {memory.get('memory', '')}\n"
                
                return f"{base_prompt}\n{memory_context}\n**Current Query:** {current_query}"
            
            return f"{base_prompt}\n\n**Current Query:** {current_query}"
        except Exception as e:
            logger.error(f"Failed to enhance prompt: {e}")
            return f"{base_prompt}\n\n**Current Query:** {current_query}"
    
    def update_memory_from_interaction(self,
                                     user_id: str,
                                     user_message: str,
                                     agent_response: str,
                                     agent_name: str,
                                     metadata: Dict[str, Any] = None):
        """Update memory from interaction."""
        try:
            if not user_message or not user_message.strip():
                return
            
            summary = f"User: {user_message[:100]}... | Agent: {agent_name}"
            self.memory.add(
                summary,
                user_id=user_id,
                metadata={**(metadata or {}), "agent_name": agent_name}
            )
        except Exception as e:
            logger.error(f"Failed to update memory: {e}")
    
    def get_contextual_memories_for_agent(self,
                                        user_id: str,
                                        agent_name: str,
                                        current_query: str,
                                        limit: int = 3) -> List[Dict[str, Any]]:
        """Get contextual memories for an agent."""
        try:
            return self.memory.search(
                f"{current_query} {agent_name}",
                user_id=user_id,
                limit=limit,
                filters={"agent_name": agent_name}
            )
        except Exception as e:
            logger.error(f"Failed to get contextual memories: {e}")
            return [] 
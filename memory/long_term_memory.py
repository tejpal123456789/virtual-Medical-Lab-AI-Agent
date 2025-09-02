"""
Long-term Memory Management System using mem0

This module provides intelligent long-term memory capabilities for the 
Multi-Agent Medical Assistant, enabling persistent learning and context retention.
"""

import os
import json
import uuid
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

from mem0 import Memory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memories to store."""
    USER_PROFILE = "user_profile"
    MEDICAL_HISTORY = "medical_history" 
    CONVERSATION_CONTEXT = "conversation_context"
    AGENT_LEARNING = "agent_learning"
    MEDICAL_INSIGHTS = "medical_insights"
    PREFERENCES = "preferences"


@dataclass
class MemoryEntry:
    """Structure for memory entries."""
    memory_id: str
    memory_type: MemoryType
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    user_id: Optional[str] = None
    agent_name: Optional[str] = None
    confidence: float = 1.0
    relevance_score: float = 0.0


class LongTermMemoryManager:
    """
    Advanced long-term memory manager using mem0 for the medical assistant.
    
    Provides intelligent memory storage, retrieval, and management capabilities
    to enhance conversation continuity and personalized assistance.
    """

    def __init__(self, config):
        """Initialize the long-term memory manager."""
        self.config = config
        
        # Initialize mem0 with Qdrant backend
        memory_config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": config.memory.collection_name,
                    "host": config.memory.qdrant_host,
                    "port": config.memory.qdrant_port
                }
            },
            "graph_store": {
                "provider": "neo4j",
                "config": {
                    "url": config.memory.neo4j_url,
                    "username": config.memory.neo4j_username,
                    "password": config.memory.neo4j_password
                }
            } if (hasattr(config.memory, 'neo4j_url') and 
                  config.memory.neo4j_url and 
                  config.memory.neo4j_url.strip()) else None,
            "llm": {
                "provider": "openai",
                "config": {
                    "model": config.memory.llm_model,
                    "temperature": 0.1,
                    "api_key": os.getenv("OPENAI_API_KEY")
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": config.memory.embedding_model,
                    "api_key": os.getenv("OPENAI_API_KEY")
                    # Note: Let mem0 handle dimensions automatically for text-embedding-3-small
                }
            }
        }
        
        # Remove graph_store if Neo4j is not configured
        if not memory_config["graph_store"]:
            del memory_config["graph_store"]
            
        try:
            self.memory = Memory.from_config(memory_config)
            logger.info("✅ Long-term memory system initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize memory system: {e}")
            # Fallback to basic configuration without advanced features
            try:
                basic_config = {
                    "vector_store": {
                        "provider": "qdrant",
                        "config": {
                            "collection_name": config.memory.collection_name,
                            "host": config.memory.qdrant_host,
                            "port": config.memory.qdrant_port
                        }
                    },
                    "llm": {
                        "provider": "openai",
                        "config": {
                            "model": config.memory.llm_model,
                            "temperature": 0.1,
                            "api_key": os.getenv("OPENAI_API_KEY")
                        }
                    },
                    "embedder": {
                        "provider": "openai",
                        "config": {
                            "model": config.memory.embedding_model,
                            "api_key": os.getenv("OPENAI_API_KEY")
                            # Note: Let mem0 handle dimensions automatically
                        }
                    }
                }
                self.memory = Memory.from_config(basic_config)
                logger.info("✅ Long-term memory system initialized with basic configuration")
            except Exception as e2:
                logger.error(f"❌ Basic memory configuration also failed: {e2}")
                # Ultimate fallback to default mem0 configuration
                self.memory = Memory()
                logger.info("⚠️ Using default mem0 configuration")
        
        self.user_sessions = {}  # Track active user sessions
        
    def generate_user_id(self, session_info: Dict[str, Any] = None) -> str:
        """Generate or retrieve user ID for session tracking."""
        if session_info and "user_id" in session_info:
            return session_info["user_id"]
        
        # For demo purposes, use a default user ID
        # In production, this would come from authentication
        return "default_user"
    
    def store_conversation_context(self, 
                                 user_id: str,
                                 messages: List[BaseMessage],
                                 agent_name: str,
                                 metadata: Dict[str, Any] = None) -> str:
        """Store conversation context for future reference."""
        try:
            # Extract conversation content
            conversation_content = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    conversation_content.append(f"User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    conversation_content.append(f"Assistant: {msg.content}")
            
            content = "\n".join(conversation_content[-4:])  # Store last 2 exchanges
            
            # Skip storing if content is empty
            if not content.strip():
                return ""
            
            # Prepare metadata
            memory_metadata = {
                "agent_name": agent_name,
                "timestamp": datetime.now().isoformat(),
                "message_count": len(messages),
                "memory_type": MemoryType.CONVERSATION_CONTEXT.value,
                **(metadata or {})
            }
            
            # Store in mem0
            memory_id = self.memory.add(
                content,
                user_id=user_id,
                metadata=memory_metadata
            )
            
            logger.info(f"✅ Stored conversation context: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"❌ Failed to store conversation context: {e}")
            return ""
    
    def store_medical_insight(self,
                            user_id: str, 
                            insight: str,
                            source_agent: str,
                            confidence: float = 1.0,
                            metadata: Dict[str, Any] = None) -> str:
        """Store medical insights and findings."""
        try:
            memory_metadata = {
                "source_agent": source_agent,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "memory_type": MemoryType.MEDICAL_INSIGHTS.value,
                **(metadata or {})
            }
            
            memory_id = self.memory.add(
                insight,
                user_id=user_id,
                metadata=memory_metadata
            )
            
            logger.info(f"✅ Stored medical insight: {memory_id}")

            
            return memory_id
            
        except Exception as e:
            logger.error(f"❌ Failed to store medical insight: {e}")
            return ""
    
    def store_user_preferences(self,
                             user_id: str,
                             preferences: Dict[str, Any]) -> str:
        """Store user preferences and settings."""
        try:
            content = f"User preferences: {json.dumps(preferences, indent=2)}"
            
            memory_metadata = {
                "timestamp": datetime.now().isoformat(),
                "memory_type": MemoryType.PREFERENCES.value,
                "preferences": preferences
            }
            
            memory_id = self.memory.add(
                content,
                user_id=user_id,
                metadata=memory_metadata
            )
            
            logger.info(f"✅ Stored user preferences: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"❌ Failed to store user preferences: {e}")
            return ""
    
    def retrieve_relevant_memories(self,
                                 user_id: str,
                                 query: str,
                                 memory_types: List[MemoryType] = None,
                                 limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant memories based on query and context."""
        try:
            # Skip if query is empty
            if not query or not query.strip():
                return []
            
            # Build search filters
            filters = {}
            if memory_types:
                filters["memory_type"] = [mt.value for mt in memory_types]
            
            # Search memories
            memories = self.memory.search(
                query=query,
                user_id=user_id,
                limit=limit,
                filters=filters if filters else None
            )
            
            # Ensure memories is a list
            if not isinstance(memories, list):
                memories = []
            
            logger.info(f"✅ Retrieved {len(memories)} relevant memories")
            return memories
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve memories: {e}")
            return []
    
    def get_user_profile_summary(self, user_id: str) -> str:
        """Get a comprehensive user profile summary."""
        try:
            # Retrieve user-related memories with multiple search terms
            try:
                identity_memories = self.memory.search(
                    query="name identity user personal",
                    user_id=user_id,
                    limit=5
                )
            except Exception as e:
                logger.warning(f"Identity search failed: {e}")
                identity_memories = []
            
            try:
                profile_memories = self.memory.search(
                    query="user profile medical history preferences",
                    user_id=user_id,
                    limit=10
                )
            except Exception as e:
                logger.warning(f"Profile search failed: {e}")
                profile_memories = []
            
            # Combine and deduplicate memories safely
            all_memories = {}
            
            # Handle identity_memories
            if isinstance(identity_memories, list):
                for memory in identity_memories:
                    if isinstance(memory, dict):
                        memory_id = memory.get("id", str(uuid.uuid4()))
                        all_memories[memory_id] = memory
                    else:
                        memory_id = str(uuid.uuid4())
                        all_memories[memory_id] = {"memory": str(memory), "metadata": {}}
            
            # Handle profile_memories
            if isinstance(profile_memories, list):
                for memory in profile_memories:
                    if isinstance(memory, dict):
                        memory_id = memory.get("id", str(uuid.uuid4()))
                        if memory_id not in all_memories:  # Avoid duplicates
                            all_memories[memory_id] = memory
                    else:
                        memory_id = str(uuid.uuid4())
                        all_memories[memory_id] = {"memory": str(memory), "metadata": {}}
            
            profile_memories = list(all_memories.values())
            
            if not profile_memories:
                return "No previous interactions found for this user."
            
            # Compile profile information
            profile_sections = []
            
            # Group memories by type
            memory_groups = {}
            for memory in profile_memories:
                # Handle both dict and string memory objects
                if isinstance(memory, dict):
                    mem_type = memory.get("metadata", {}).get("memory_type", "general")
                    memory_content = memory.get("memory", str(memory))
                else:
                    mem_type = "general"
                    memory_content = str(memory)
                
                if mem_type not in memory_groups:
                    memory_groups[mem_type] = []
                memory_groups[mem_type].append(memory_content)
            
            # Build profile summary
            if "user_profile" in memory_groups:
                profile_sections.append("**User Profile:**")
                for memory_content in memory_groups["user_profile"][:3]:
                    profile_sections.append(f"- {memory_content}")
            
            if "medical_history" in memory_groups:
                profile_sections.append("\n**Medical History:**")
                for memory_content in memory_groups["medical_history"][:3]:
                    profile_sections.append(f"- {memory_content}")
            
            if "preferences" in memory_groups:
                profile_sections.append("\n**User Preferences:**")
                for memory_content in memory_groups["preferences"][:2]:
                    profile_sections.append(f"- {memory_content}")
            
            return "\n".join(profile_sections) if profile_sections else "Limited profile information available."
            
        except Exception as e:
            logger.error(f"❌ Failed to get user profile: {e}")
            return "Unable to retrieve user profile information."
    
    def enhance_prompt_with_memory(self,
                                 user_id: str,
                                 current_query: str,
                                 agent_name: str,
                                 base_prompt: str) -> str:
        """Enhance agent prompts with relevant long-term memory context."""
        try:
            # Retrieve relevant memories for current query
            relevant_memories = self.retrieve_relevant_memories(
                user_id=user_id,
                query=current_query,
                limit=3
            )
            
            # Get user profile summary
            profile_summary = self.get_user_profile_summary(user_id)
            
            # Build memory context
            memory_context = []
            
            if profile_summary and profile_summary != "No previous interactions found for this user.":
                memory_context.append(f"**User Context:**\n{profile_summary}")
            
            if relevant_memories:
                memory_context.append("\n**Relevant Previous Interactions:**")
                for i, memory in enumerate(relevant_memories[:3], 1):
                    # Handle both dict and string memory objects
                    if isinstance(memory, dict):
                        memory_content = memory.get("memory", str(memory))
                        timestamp = memory.get("metadata", {}).get("timestamp", "Unknown time")
                    else:
                        memory_content = str(memory)
                        timestamp = "Unknown time"
                    
                    memory_context.append(f"{i}. {memory_content} (from {timestamp})")
            
            # Enhance the base prompt
            if memory_context:
                enhanced_prompt = f"""{base_prompt}

**LONG-TERM MEMORY CONTEXT:**
{chr(10).join(memory_context)}

**INSTRUCTIONS FOR USING MEMORY:**
- Reference relevant previous interactions when appropriate
- Maintain consistency with user's known preferences and medical history
- Build upon previous conversations naturally
- If user mentions something that contradicts stored information, clarify politely

**CURRENT QUERY:** {current_query}
"""
            else:
                enhanced_prompt = f"{base_prompt}\n\n**CURRENT QUERY:** {current_query}"
            
            logger.info(f"✅ Enhanced prompt with {len(relevant_memories)} memories for {agent_name}")
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"❌ Failed to enhance prompt with memory: {e}")
            return f"{base_prompt}\n\n**CURRENT QUERY:** {current_query}"
    
    def update_memory_from_interaction(self,
                                     user_id: str,
                                     user_message: str,
                                     agent_response: str,
                                     agent_name: str,
                                     metadata: Dict[str, Any] = None):
        """Update long-term memory based on the interaction."""
        try:
            # Skip if user_message or agent_response is empty
            if not user_message or not user_message.strip():
                return
                
            if not agent_response or not agent_response.strip():
                return
            
            # Extract and store key information from the interaction
            # Check if this contains personal identity information (avoid questions)
            user_lower = user_message.lower()
            is_identity_statement = (
                user_lower.startswith("my name is") or
                user_lower.startswith("i am") or
                user_lower.startswith("call me") or
                " my name is " in user_lower
            )
            
            # Make sure it's not a question
            is_question = any(q in user_lower for q in [
                "what is", "what's", "do you know", "can you tell me", "remember my"
            ])
            
            if is_identity_statement and not is_question:
                # Extract name from the message
                if "my name is" in user_lower:
                    name_start = user_lower.find("my name is") + 10
                    name_part = user_message[name_start:].strip()
                    interaction_summary = f"User's name: {name_part}"
                elif user_lower.startswith("i am"):
                    name_part = user_message[4:].strip()
                    interaction_summary = f"User's name: {name_part}"
                elif user_lower.startswith("call me"):
                    name_part = user_message[7:].strip()
                    interaction_summary = f"User prefers to be called: {name_part}"
                else:
                    interaction_summary = f"User identity info: {user_message}"
            else:
                interaction_summary = f"User asked: {user_message[:200]}...\nAssistant ({agent_name}) responded."
            
            # Store the interaction
            interaction_metadata = {
                "agent_name": agent_name,
                "timestamp": datetime.now().isoformat(),
                "memory_type": MemoryType.CONVERSATION_CONTEXT.value,
                "interaction_type": "query_response",
                **(metadata or {})
            }
            
            self.memory.add(
                interaction_summary,
                user_id=user_id,
                metadata=interaction_metadata
            )
            
            # Extract and store medical insights if relevant
            if any(term in user_message.lower() for term in [
                "symptom", "diagnosis", "treatment", "medication", "condition", 
                "disease", "pain", "health", "medical", "doctor", "hospital"
            ]):
                medical_context = f"Medical query about: {user_message}"
                medical_metadata = {
                    "agent_name": agent_name,
                    "timestamp": datetime.now().isoformat(),
                    "memory_type": MemoryType.MEDICAL_INSIGHTS.value,
                    "query_type": "medical_question"
                }
                
                self.memory.add(
                    medical_context,
                    user_id=user_id,
                    metadata=medical_metadata
                )
            
            logger.info(f"✅ Updated long-term memory for user {user_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to update memory: {e}")
    
    def get_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about stored memories for a user."""
        try:
            all_memories = self.memory.get_all(user_id=user_id)
            
            stats = {
                "total_memories": len(all_memories),
                "memory_types": {},
                "date_range": {},
                "agent_interactions": {}
            }
            
            for memory in all_memories:
                metadata = memory.get("metadata", {})
                mem_type = metadata.get("memory_type", "unknown")
                agent = metadata.get("agent_name", "unknown")
                timestamp = metadata.get("timestamp")
                
                # Count by type
                stats["memory_types"][mem_type] = stats["memory_types"].get(mem_type, 0) + 1
                
                # Count by agent
                stats["agent_interactions"][agent] = stats["agent_interactions"].get(agent, 0) + 1
                
                # Track date range
                if timestamp:
                    if not stats["date_range"]:
                        stats["date_range"]["earliest"] = timestamp
                        stats["date_range"]["latest"] = timestamp
                    else:
                        if timestamp < stats["date_range"]["earliest"]:
                            stats["date_range"]["earliest"] = timestamp
                        if timestamp > stats["date_range"]["latest"]:
                            stats["date_range"]["latest"] = timestamp
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get memory stats: {e}")
            return {"total_memories": 0, "error": str(e)}
    
    def cleanup_old_memories(self, user_id: str, days_to_keep: int = 30):
        """Clean up old conversation memories while keeping important medical insights."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            all_memories = self.memory.get_all(user_id=user_id)
            
            deleted_count = 0
            for memory in all_memories:
                metadata = memory.get("metadata", {})
                timestamp_str = metadata.get("timestamp")
                memory_type = metadata.get("memory_type")
                
                if timestamp_str:
                    memory_date = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00").replace("+00:00", ""))
                    
                    # Only delete old conversation contexts, keep medical insights
                    if (memory_date < cutoff_date and 
                        memory_type == MemoryType.CONVERSATION_CONTEXT.value):
                        
                        self.memory.delete(memory["id"])
                        deleted_count += 1
            
            logger.info(f"✅ Cleaned up {deleted_count} old conversation memories")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old memories: {e}")
            return 0
    
    def export_user_memories(self, user_id: str) -> Dict[str, Any]:
        """Export all memories for a user (for backup or analysis)."""
        try:
            all_memories = self.memory.get_all(user_id=user_id)
            
            export_data = {
                "user_id": user_id,
                "export_timestamp": datetime.now().isoformat(),
                "total_memories": len(all_memories),
                "memories": all_memories,
                "stats": self.get_memory_stats(user_id)
            }
            
            logger.info(f"✅ Exported {len(all_memories)} memories for user {user_id}")
            return export_data
            
        except Exception as e:
            logger.error(f"❌ Failed to export memories: {e}")
            return {"error": str(e)}
    
    def get_contextual_memories_for_agent(self,
                                        user_id: str,
                                        agent_name: str,
                                        current_query: str,
                                        limit: int = 3) -> List[Dict[str, Any]]:
        """Get memories most relevant to the current agent and query."""
        try:
            # Search for memories relevant to this agent and query
            agent_memories = self.memory.search(
                query=f"{current_query} {agent_name}",
                user_id=user_id,
                limit=limit,
                filters={"agent_name": agent_name}
            )
            
            # Also get general relevant memories
            general_memories = self.memory.search(
                query=current_query,
                user_id=user_id,
                limit=limit//2 + 1
            )
            
            # Combine and deduplicate
            all_memories = {}
            for memory in agent_memories + general_memories:
                # Handle both dict and string memory objects
                if isinstance(memory, dict):
                    memory_id = memory.get("id", str(uuid.uuid4()))
                    all_memories[memory_id] = memory
                else:
                    # For string memories, create a basic dict structure
                    memory_id = str(uuid.uuid4())
                    all_memories[memory_id] = {
                        "id": memory_id,
                        "memory": str(memory),
                        "score": 0.5,
                        "metadata": {}
                    }
            
            # Sort by relevance score
            sorted_memories = sorted(
                all_memories.values(),
                key=lambda x: x.get("score", 0) if isinstance(x, dict) else 0,
                reverse=True
            )
            
            return sorted_memories[:limit]
            
        except Exception as e:
            logger.error(f"❌ Failed to get contextual memories: {e}")
            return []


class MemoryEnhancedPromptBuilder:
    """Helper class to build memory-enhanced prompts for different agents."""
    
    def __init__(self, memory_manager: LongTermMemoryManager):
        self.memory_manager = memory_manager
    
    def enhance_conversation_prompt(self, 
                                  user_id: str,
                                  query: str,
                                  base_prompt: str) -> str:
        """Enhance conversation agent prompt with memory context."""
        
        # Special handling for identity-related queries
        if any(keyword in query.lower() for keyword in [
            "what is my name", "my name", "who am i", "what do you know about me",
            "do you remember me", "my identity"
        ]):
            # Specifically retrieve identity and personal information
            identity_memories = self.memory_manager.retrieve_relevant_memories(
                user_id=user_id,
                query="name identity personal info",
                limit=3
            )
            
            if identity_memories:
                identity_context = "\n**What I know about you:**\n"
                for i, memory in enumerate(identity_memories, 1):
                    if isinstance(memory, dict):
                        memory_content = memory.get("memory", str(memory))
                    else:
                        memory_content = str(memory)
                    identity_context += f"- {memory_content}\n"
                
                enhanced_prompt = f"""{base_prompt}

{identity_context}

**Current Query:** {query}

Please use the information above to answer the user's question about their identity or personal information.
"""
                return enhanced_prompt
        
        # Default memory enhancement
        return self.memory_manager.enhance_prompt_with_memory(
            user_id=user_id,
            current_query=query,
            agent_name="CONVERSATION_AGENT", 
            base_prompt=base_prompt
        )
    
    def enhance_rag_prompt(self,
                          user_id: str, 
                          query: str,
                          base_prompt: str) -> str:
        """Enhance RAG agent prompt with relevant medical history."""
        relevant_memories = self.memory_manager.get_contextual_memories_for_agent(
            user_id=user_id,
            agent_name="RAG_AGENT",
            current_query=query,
            limit=2
        )
        
        memory_context = ""
        if relevant_memories:
            memory_context = "\n**Relevant Medical History from Previous Interactions:**\n"
            for i, memory in enumerate(relevant_memories, 1):
                # Handle both dict and string memory objects
                if isinstance(memory, dict):
                    memory_content = memory.get("memory", str(memory))
                else:
                    memory_content = str(memory)
                memory_context += f"{i}. {memory_content}\n"
        
        enhanced_prompt = f"""{base_prompt}

{memory_context}

**Current Medical Query:** {query}

Please consider the user's medical history when providing information, and ensure consistency with previous medical discussions.
"""
        return enhanced_prompt
    
    def enhance_medical_research_prompt(self,
                                      user_id: str,
                                      query: str, 
                                      base_prompt: str) -> str:
        """Enhance medical research agent prompt with research history."""
        research_memories = self.memory_manager.retrieve_relevant_memories(
            user_id=user_id,
            query=query,
            memory_types=[MemoryType.MEDICAL_INSIGHTS, MemoryType.MEDICAL_HISTORY],
            limit=3
        )
        
        research_context = ""
        if research_memories:
            research_context = "\n**Previous Medical Research Context:**\n"
            for i, memory in enumerate(research_memories, 1):
                # Handle both dict and string memory objects
                if isinstance(memory, dict):
                    memory_content = memory.get("memory", str(memory))
                else:
                    memory_content = str(memory)
                research_context += f"{i}. {memory_content}\n"
        
        enhanced_prompt = f"""{base_prompt}

{research_context}

**Current Research Query:** {query}

Build upon previous research context when relevant and avoid duplicating information already provided to this user.
"""
        return enhanced_prompt 
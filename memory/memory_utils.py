"""
Utility functions for memory management and maintenance.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import os

logger = logging.getLogger(__name__)


def extract_medical_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract medical entities from text for better memory categorization.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary with categorized medical entities
    """
    medical_keywords = {
        "symptoms": [
            "pain", "headache", "fever", "cough", "nausea", "fatigue", "dizziness",
            "chest pain", "shortness of breath", "rash", "swelling", "bleeding"
        ],
        "conditions": [
            "diabetes", "hypertension", "covid", "cancer", "tumor", "infection",
            "pneumonia", "asthma", "arthritis", "depression", "anxiety"
        ],
        "treatments": [
            "medication", "surgery", "therapy", "treatment", "prescription",
            "antibiotics", "chemotherapy", "radiation", "physiotherapy"
        ],
        "body_parts": [
            "heart", "lung", "brain", "liver", "kidney", "stomach", "skin",
            "chest", "head", "back", "arm", "leg", "eye", "ear"
        ]
    }
    
    found_entities = {category: [] for category in medical_keywords.keys()}
    text_lower = text.lower()
    
    for category, keywords in medical_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                found_entities[category].append(keyword)
    
    # Remove empty categories
    return {k: v for k, v in found_entities.items() if v}


def should_store_as_medical_history(user_message: str, agent_response: str) -> bool:
    """
    Determine if an interaction should be stored as medical history.
    
    Args:
        user_message: User's message
        agent_response: Agent's response
        
    Returns:
        Boolean indicating if this should be stored as medical history
    """
    medical_indicators = [
        "symptoms", "diagnosis", "treatment", "medication", "condition",
        "medical history", "family history", "allergies", "chronic",
        "prescription", "doctor", "hospital", "clinic", "surgery"
    ]
    
    combined_text = f"{user_message} {agent_response}".lower()
    
    return any(indicator in combined_text for indicator in medical_indicators)


def extract_user_preferences(messages: List[BaseMessage]) -> Dict[str, Any]:
    """
    Extract user preferences from conversation history.
    
    Args:
        messages: List of conversation messages
        
    Returns:
        Dictionary of extracted preferences
    """
    preferences = {
        "communication_style": "standard",
        "detail_level": "moderate",
        "language": "english",
        "medical_focus_areas": [],
        "preferred_agents": []
    }
    
    # Analyze messages for preference indicators
    for message in messages:
        if isinstance(message, HumanMessage):
            content = message.content.lower()
            
            # Communication style preferences
            if any(phrase in content for phrase in ["simple terms", "easy to understand", "layman"]):
                preferences["communication_style"] = "simple"
            elif any(phrase in content for phrase in ["technical", "detailed", "medical terminology"]):
                preferences["communication_style"] = "technical"
            
            # Detail level preferences
            if any(phrase in content for phrase in ["brief", "quick", "short"]):
                preferences["detail_level"] = "brief"
            elif any(phrase in content for phrase in ["detailed", "comprehensive", "thorough"]):
                preferences["detail_level"] = "detailed"
            
            # Medical focus areas
            focus_areas = extract_medical_entities(content)
            for category, entities in focus_areas.items():
                preferences["medical_focus_areas"].extend(entities)
    
    # Remove duplicates from medical focus areas
    preferences["medical_focus_areas"] = list(set(preferences["medical_focus_areas"]))
    
    return preferences


def format_memory_for_prompt(memories: List[Dict[str, Any]], max_length: int = 500) -> str:
    """
    Format memories for inclusion in prompts with length constraints.
    
    Args:
        memories: List of memory objects
        max_length: Maximum length of formatted text
        
    Returns:
        Formatted memory text for prompt inclusion
    """
    if not memories:
        return ""
    
    formatted_lines = []
    current_length = 0
    
    for i, memory in enumerate(memories):
        # Handle both dict and string memory objects
        if isinstance(memory, dict):
            memory_text = memory.get("memory", str(memory))
            timestamp = memory.get("metadata", {}).get("timestamp", "")
        else:
            memory_text = str(memory)
            timestamp = ""
        
        # Format timestamp
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00").replace("+00:00", ""))
                time_str = dt.strftime("%m/%d")
            except:
                time_str = "recent"
        else:
            time_str = "recent"
        
        line = f"• ({time_str}) {memory_text}"
        
        # Check if adding this line would exceed max length
        if current_length + len(line) > max_length:
            if i == 0:  # Always include at least one memory
                formatted_lines.append(line[:max_length-10] + "...")
            break
        
        formatted_lines.append(line)
        current_length += len(line)
    
    return "\n".join(formatted_lines)


def create_memory_backup(memory_manager, user_id: str, backup_path: str = None) -> str:
    """
    Create a backup of user's memories.
    
    Args:
        memory_manager: LongTermMemoryManager instance
        user_id: User ID to backup
        backup_path: Optional custom backup path
        
    Returns:
        Path to the backup file
    """
    try:
        if not backup_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"./memory/backups/user_{user_id}_{timestamp}.json"
        
        # Ensure backup directory exists
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        
        # Export memories
        export_data = memory_manager.export_user_memories(user_id)
        
        # Save to file
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Created memory backup: {backup_path}")
        return backup_path
        
    except Exception as e:
        logger.error(f"❌ Failed to create memory backup: {e}")
        return ""


def validate_memory_system(memory_manager) -> Dict[str, Any]:
    """
    Validate the memory system configuration and connectivity.
    
    Args:
        memory_manager: LongTermMemoryManager instance
        
    Returns:
        Validation results
    """
    validation_results = {
        "memory_system_status": "unknown",
        "tests_passed": 0,
        "tests_failed": 0,
        "errors": []
    }
    
    try:
        # Test 1: Basic memory operations
        test_user_id = "test_user_validation"
        test_content = "This is a test memory entry for system validation."
        
        # Add test memory
        try:
            memory_id = memory_manager.memory.add(
                test_content,
                user_id=test_user_id,
                metadata={"test": True, "timestamp": datetime.now().isoformat()}
            )
        except Exception as e:
            validation_results["errors"].append(f"Failed to add memory: {str(e)}")
            memory_id = None
        
        if memory_id:
            validation_results["tests_passed"] += 1
            
            # Test 2: Search functionality
            search_results = memory_manager.memory.search(
                query="test memory validation",
                user_id=test_user_id,
                limit=1
            )
            
            if search_results:
                validation_results["tests_passed"] += 1
            else:
                validation_results["tests_failed"] += 1
                validation_results["errors"].append("Memory search functionality failed")
            
            # Test 3: Delete functionality  
            try:
                memory_manager.memory.delete(memory_id)
                validation_results["tests_passed"] += 1
            except:
                validation_results["tests_failed"] += 1
                validation_results["errors"].append("Memory deletion functionality failed")
        else:
            validation_results["tests_failed"] += 1
            validation_results["errors"].append("Memory addition functionality failed")
        
        # Determine overall status
        if validation_results["tests_failed"] == 0:
            validation_results["memory_system_status"] = "healthy"
        elif validation_results["tests_passed"] > validation_results["tests_failed"]:
            validation_results["memory_system_status"] = "partially_functional"
        else:
            validation_results["memory_system_status"] = "failed"
            
    except Exception as e:
        validation_results["memory_system_status"] = "error"
        validation_results["errors"].append(f"System validation error: {str(e)}")
        validation_results["tests_failed"] += 1
    
    return validation_results 
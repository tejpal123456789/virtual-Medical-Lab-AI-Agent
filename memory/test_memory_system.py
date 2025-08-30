"""
Test script for long-term memory system validation.

This script tests the memory functionality to ensure proper integration
with the multi-agent medical assistant system.
"""

import sys
import os
import asyncio
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from memory import LongTermMemoryManager, MemoryEnhancedPromptBuilder
from memory.memory_utils import validate_memory_system, extract_medical_entities


def test_memory_system():
    """Comprehensive test of the long-term memory system."""
    print("🧪 Testing Long-term Memory System...")
    
    try:
        # Load configuration
        config = Config()
        print("✅ Configuration loaded")
        
        # Initialize memory manager
        memory_manager = LongTermMemoryManager(config)
        print("✅ Memory manager initialized")
        
        # Initialize prompt builder
        prompt_builder = MemoryEnhancedPromptBuilder(memory_manager)
        print("✅ Prompt builder initialized")
        
        # Test user ID generation
        test_user_id = memory_manager.generate_user_id()
        print(f"✅ Generated user ID: {test_user_id}")
        
        # Test basic memory operations
        print("\n🔍 Testing basic memory operations...")
        
        # Test 1: Store conversation context
        test_memory_id = memory_manager.store_conversation_context(
            user_id=test_user_id,
            messages=[],  # Empty for test
            agent_name="TEST_AGENT",
            metadata={"test_type": "unit_test"}
        )
        print(f"✅ Stored test conversation: {test_memory_id}")
        
        # Test 2: Store medical insight
        insight_id = memory_manager.store_medical_insight(
            user_id=test_user_id,
            insight="Patient reported headache symptoms during consultation",
            source_agent="CONVERSATION_AGENT",
            confidence=0.9
        )
        print(f"✅ Stored medical insight: {insight_id}")
        
        # Test 3: Store user preferences
        test_preferences = {
            "communication_style": "simple",
            "detail_level": "moderate",
            "medical_focus": ["headaches", "general_health"]
        }
        pref_id = memory_manager.store_user_preferences(
            user_id=test_user_id,
            preferences=test_preferences
        )
        print(f"✅ Stored user preferences: {pref_id}")
        
        # Test 4: Retrieve memories
        print("\n🔍 Testing memory retrieval...")
        relevant_memories = memory_manager.retrieve_relevant_memories(
            user_id=test_user_id,
            query="headache symptoms",
            limit=3
        )
        print(f"✅ Retrieved {len(relevant_memories)} relevant memories")
        
        # Test 5: Get user profile
        profile = memory_manager.get_user_profile_summary(test_user_id)
        print(f"✅ Retrieved user profile: {profile[:100]}...")
        
        # Test 6: Memory statistics
        stats = memory_manager.get_memory_stats(test_user_id)
        print(f"✅ Memory stats: {stats}")
        
        # Test 7: Prompt enhancement
        test_prompt = "You are a medical assistant. Please answer the user's question."
        enhanced_prompt = prompt_builder.enhance_conversation_prompt(
            user_id=test_user_id,
            query="I have a headache again",
            base_prompt=test_prompt
        )
        print(f"✅ Enhanced prompt length: {len(enhanced_prompt)} characters")
        
        # Test 8: System validation
        print("\n🔍 Running system validation...")
        validation_results = validate_memory_system(memory_manager)
        print(f"✅ System validation: {validation_results}")
        
        # Test 9: Medical entity extraction
        test_text = "I have been experiencing chest pain and shortness of breath"
        entities = extract_medical_entities(test_text)
        print(f"✅ Extracted medical entities: {entities}")
        
        # Cleanup test data
        print("\n🧹 Cleaning up test data...")
        try:
            if test_memory_id:
                memory_manager.memory.delete(test_memory_id)
            if insight_id:
                memory_manager.memory.delete(insight_id)  
            if pref_id:
                memory_manager.memory.delete(pref_id)
            print("✅ Test data cleaned up")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")
        
        print("\n🎉 All memory system tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Memory system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_integration():
    """Test memory integration with agent decision system."""
    print("\n🔧 Testing Memory Integration...")
    
    try:
        # Import agent decision components
        from agents.agent_decision import create_agent_graph, init_agent_state
        
        # Create the graph
        graph = create_agent_graph()
        print("✅ Agent graph created with memory integration")
        
        # Initialize state
        state = init_agent_state()
        state["current_input"] = "Hello, I'm testing the memory system"
        
        # Test the workflow
        result = graph.invoke(state, {"configurable": {"thread_id": "test_thread"}})
        print("✅ Memory-enhanced workflow executed successfully")
        
        # Check if memory fields are present
        if "memory_enhanced" in result:
            print(f"✅ Memory enhancement status: {result['memory_enhanced']}")
        
        if "user_id" in result:
            print(f"✅ User ID tracked: {result['user_id']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Starting Memory System Tests...")
    print("=" * 50)
    
    # Test basic memory functionality
    basic_test_passed = test_memory_system()
    
    # Test integration with agent system  
    integration_test_passed = test_memory_integration()
    
    print("\n" + "=" * 50)
    if basic_test_passed and integration_test_passed:
        print("🎉 ALL TESTS PASSED! Memory system is ready for use.")
    else:
        print("❌ Some tests failed. Please check the configuration.")
        
    print("Memory system testing completed.") 
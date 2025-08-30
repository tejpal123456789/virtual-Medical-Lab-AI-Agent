"""
Test script to verify name/identity memory storage and retrieval.
"""

import os
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def test_name_memory_flow():
    """Test the complete flow of storing and retrieving user identity."""
    print("🧪 Testing Name Memory Flow...")
    
    try:
        from config import Config
        from memory import LongTermMemoryManager
        
        config = Config()
        memory_manager = LongTermMemoryManager(config)
        
        test_user_id = "tejpal_test_user"
        
        print("1. Storing user name information...")
        
        # Store name as the system would when user says "my name is Tejpal Kumawat"
        name_info = "User introduced themselves: my name is Tejpal Kumawat"
        memory_id = memory_manager.store_medical_insight(
            user_id=test_user_id,
            insight=name_info,
            source_agent="CONVERSATION_AGENT",
            confidence=1.0,
            metadata={
                "info_type": "user_identity", 
                "immediate_storage": True
            }
        )
        print(f"✅ Stored name info with ID: {memory_id}")
        
        # Also store as regular interaction
        memory_manager.update_memory_from_interaction(
            user_id=test_user_id,
            user_message="my name is Tejpal Kumawat",
            agent_response="Nice to meet you, Tejpal!",
            agent_name="CONVERSATION_AGENT"
        )
        print("✅ Stored regular interaction")
        
        # Small delay to ensure storage is complete
        time.sleep(1)
        
        print("\n2. Testing name retrieval...")
        
        # Test different search queries
        search_queries = [
            "name identity user personal",
            "Tejpal Kumawat",
            "user name",
            "my name"
        ]
        
        for query in search_queries:
            memories = memory_manager.retrieve_relevant_memories(
                user_id=test_user_id,
                query=query,
                limit=5
            )
            print(f"✅ Search '{query}': Found {len(memories)} memories")
            for memory in memories:
                if isinstance(memory, dict):
                    content = memory.get("memory", str(memory))
                else:
                    content = str(memory)
                print(f"   - {content[:50]}...")
        
        print("\n3. Testing user profile summary...")
        profile = memory_manager.get_user_profile_summary(test_user_id)
        print(f"✅ Profile summary: {profile}")
        
        print("\n4. Testing identity-specific retrieval...")
        identity_memories = memory_manager.retrieve_relevant_memories(
            user_id=test_user_id,
            query="name identity personal info", 
            limit=3
        )
        print(f"✅ Identity memories: {len(identity_memories)} found")
        for memory in identity_memories:
            if isinstance(memory, dict):
                content = memory.get("memory", str(memory))
            else:
                content = str(memory)
            print(f"   - {content}")
        
        # Clean up test data
        print("\n5. Cleaning up test data...")
        all_memories = memory_manager.memory.get_all(user_id=test_user_id)
        for memory in all_memories:
            try:
                if isinstance(memory, dict) and "id" in memory:
                    memory_manager.memory.delete(memory["id"])
            except:
                pass
        print("✅ Test data cleaned up")
        
        print("\n🎉 Name memory flow test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Name memory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_integration():
    """Test how the agent handles name queries."""
    print("\n🧪 Testing Agent Integration with Name Memory...")
    
    try:
        from agents.agent_decision import process_query
        
        test_session_id = "tejpal_session_test"
        
        print("1. First interaction - introducing name...")
        result1 = process_query("my name is Tejpal Kumawat", session_id=test_session_id)
        print("✅ First response received")
        
        # Small delay to ensure memory is stored
        time.sleep(2)
        
        print("\n2. Second interaction - asking about name...")
        result2 = process_query("what is my name?", session_id=test_session_id)
        response_text = result2['messages'][-1].content
        print(f"✅ Second response: {response_text[:100]}...")
        
        # Check if the response contains the name
        if "tejpal" in response_text.lower() or "kumawat" in response_text.lower():
            print("🎉 SUCCESS! The system remembered the user's name!")
            return True
        else:
            print("⚠️ The system didn't remember the name correctly")
            print(f"Full response: {response_text}")
            return False
            
    except Exception as e:
        print(f"❌ Agent integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Testing Name Memory System")
    print("=" * 50)
    
    # Test 1: Basic name memory operations
    basic_test = test_name_memory_flow()
    
    # Test 2: Full agent integration
    agent_test = test_agent_integration()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"Basic Name Memory: {'✅ PASS' if basic_test else '❌ FAIL'}")
    print(f"Agent Integration: {'✅ PASS' if agent_test else '❌ FAIL'}")
    
    if basic_test and agent_test:
        print("\n🎉 Name memory system is working correctly!")
        print("Now when you say 'my name is [Name]' and later ask 'what is my name?',")
        print("the system will remember and tell you your name.")
    else:
        print("\n⚠️ Some tests failed. The memory system needs more work.") 
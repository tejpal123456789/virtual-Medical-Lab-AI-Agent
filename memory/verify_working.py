"""
Verification script to confirm the memory system is working correctly.
"""

import os
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def verify_memory_working():
    """Verify the memory system works with the exact scenario the user experienced."""
    print("🔍 Verifying Memory System with User Scenario")
    print("=" * 50)
    
    try:
        from agents.agent_decision import process_query
        
        # Use a test session ID
        test_session = "verify_memory_test"
        
        print("Step 1: User says 'my name is Tejpal'...")
        result1 = process_query("my name is Tejpal", session_id=test_session)
        response1 = result1['messages'][-1].content
        print(f"✅ Response 1: {response1}")
        
        # Small delay to ensure memory processing
        time.sleep(2)
        
        print("\nStep 2: User asks 'what is my name'...")
        result2 = process_query("what is my name", session_id=test_session)
        response2 = result2['messages'][-1].content
        print(f"✅ Response 2: {response2}")
        
        # Check if the name is remembered
        if "tejpal" in response2.lower():
            print("\n🎉 SUCCESS! Memory system is working correctly!")
            print("✅ The system correctly remembered the user's name")
            return True
        else:
            print("\n⚠️ Memory system may not be working correctly")
            print(f"Expected to see 'Tejpal' in response, got: {response2}")
            return False
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_memory_storage():
    """Check if memories are actually being stored."""
    print("\n🔍 Checking Memory Storage...")
    
    try:
        from config import Config
        from memory import LongTermMemoryManager
        
        config = Config()
        memory_manager = LongTermMemoryManager(config)
        
        test_user = "storage_check_user"
        
        # Store a test memory
        print("1. Storing test memory...")
        memory_id = memory_manager.store_medical_insight(
            user_id=test_user,
            insight="Test user name: John Doe",
            source_agent="TEST_AGENT",
            confidence=1.0,
            metadata={"info_type": "user_identity"}
        )
        print(f"✅ Memory stored with ID: {memory_id}")
        
        # Try to retrieve it
        print("2. Retrieving test memory...")
        memories = memory_manager.retrieve_relevant_memories(
            user_id=test_user,
            query="John Doe name",
            limit=5
        )
        print(f"✅ Retrieved {len(memories)} memories")
        
        for memory in memories:
            if isinstance(memory, dict):
                content = memory.get("memory", str(memory))
            else:
                content = str(memory)
            print(f"   - {content}")
        
        # Get all memories for this user
        print("3. Getting all memories...")
        all_memories = memory_manager.memory.get_all(user_id=test_user)
        print(f"✅ Total memories for user: {len(all_memories)}")
        
        # Clean up
        print("4. Cleaning up...")
        for memory in all_memories:
            try:
                if isinstance(memory, dict) and "id" in memory:
                    memory_manager.memory.delete(memory["id"])
            except Exception as e:
                print(f"⚠️ Cleanup warning: {e}")
        
        print("✅ Storage check completed!")
        return len(memories) > 0
        
    except Exception as e:
        print(f"❌ Storage check failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Memory System Verification")
    print("Testing the exact scenario: 'my name is Tejpal' → 'what is my name'")
    print("=" * 70)
    
    # Test 1: Check basic storage
    storage_works = check_memory_storage()
    
    # Test 2: Full workflow test
    workflow_works = verify_memory_working()
    
    print("\n" + "=" * 70)
    print("📊 Verification Results:")
    print(f"Memory Storage: {'✅ WORKING' if storage_works else '❌ FAILED'}")
    print(f"Name Memory Flow: {'✅ WORKING' if workflow_works else '❌ FAILED'}")
    
    if storage_works and workflow_works:
        print("\n🎉 VERIFICATION PASSED!")
        print("Your memory system is working correctly.")
        print("The user can now introduce themselves and the system will remember!")
    else:
        print("\n⚠️ VERIFICATION FAILED!")
        print("There are still issues that need to be resolved.")
    
    print("\n📝 Note: Based on your logs, the system actually IS working!")
    print("The response 'Your name is Tejpal!' shows memory is functioning.") 
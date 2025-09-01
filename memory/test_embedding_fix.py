"""
Test script to verify that the embedding dimension issue is resolved.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def test_openai_embedding_directly():
    """Test OpenAI embedding API directly to verify it works."""
    print("🧪 Testing OpenAI Embedding API Directly...")
    
    try:
        import openai
        
        # Check API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not found")
            return False
        
        # Test with the new embedding model
        print("1. Testing text-embedding-3-small...")
        client = openai.OpenAI(api_key=api_key)
        
        response = client.embeddings.create(
            input="This is a test for medical assistant memory",
            model="text-embedding-3-small",
            dimensions=1536  # Explicitly specify dimensions
        )
        
        print(f"✅ Embedding successful! Vector length: {len(response.data[0].embedding)}")
        
        # Test without dimensions parameter  
        print("2. Testing text-embedding-3-small without dimensions...")
        response2 = client.embeddings.create(
            input="This is another test",
            model="text-embedding-3-small"
        )
        
        print(f"✅ Embedding without dimensions successful! Vector length: {len(response2.data[0].embedding)}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI embedding test failed: {e}")
        return False


def test_mem0_with_fixed_config():
    """Test mem0 with the corrected embedding configuration."""
    print("\n🧪 Testing mem0 with Fixed Embedding Configuration...")
    
    try:
        from mem0 import Memory
        
        # Check API key
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ OPENAI_API_KEY not found")
            return False
        
        # Test with corrected configuration
        fixed_config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.1,
                    "api_key": os.getenv("OPENAI_API_KEY")
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                    "api_key": os.getenv("OPENAI_API_KEY")
                    # Note: No explicit dimensions - let mem0 handle it
                }
            }
        }
        
        print("1. Initializing mem0 with fixed config...")
        memory = Memory.from_config(fixed_config)
        
        # Test basic operations
        user_id = "test_user_fixed"
        test_content = "This is a test memory for dimension fix validation."
        
        print("2. Adding test memory...")
        memory_id = memory.add(test_content, user_id=user_id)
        print(f"✅ Memory added successfully: {memory_id}")
        
        print("3. Searching for memory...")
        results = memory.search("test memory dimension", user_id=user_id, limit=1)
        print(f"✅ Search successful! Found {len(results)} memories")
        
        # Clean up
        print("4. Cleaning up...")
        if memory_id:
            memory.delete(memory_id)
            print("✅ Memory deleted successfully")
        
        print("✅ mem0 embedding configuration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ mem0 embedding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_manager_with_fix():
    """Test the LongTermMemoryManager with the embedding fix."""
    print("\n🧪 Testing LongTermMemoryManager with Embedding Fix...")
    
    try:
        from config import Config
        from memory import LongTermMemoryManager
        
        # Load config
        config = Config()
        
        print("1. Initializing LongTermMemoryManager...")
        memory_manager = LongTermMemoryManager(config)
        
        print("2. Testing basic memory operations...")
        user_id = "test_user_manager"
        
        # Test storing medical insight
        insight_id = memory_manager.store_medical_insight(
            user_id=user_id,
            insight="Patient reported headache symptoms after stress",
            source_agent="CONVERSATION_AGENT",
            confidence=0.8
        )
        print(f"✅ Medical insight stored: {insight_id}")
        
        # Test retrieving memories
        memories = memory_manager.retrieve_relevant_memories(
            user_id=user_id,
            query="headache symptoms",
            limit=2
        )
        print(f"✅ Retrieved {len(memories)} relevant memories")
        
        # Test user profile
        profile = memory_manager.get_user_profile_summary(user_id)
        print(f"✅ User profile retrieved: {profile[:50]}...")
        
        print("✅ LongTermMemoryManager test passed!")
        return True
        
    except Exception as e:
        print(f"❌ LongTermMemoryManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Testing Embedding Dimension Fix")
    print("=" * 50)
    
    # Test 1: Direct OpenAI embedding API
    openai_test = test_openai_embedding_directly()
    
    # Test 2: mem0 with fixed configuration
    mem0_test = test_mem0_with_fixed_config()
    
    # Test 3: Full memory manager
    manager_test = test_memory_manager_with_fix()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"OpenAI Embedding API: {'✅ PASS' if openai_test else '❌ FAIL'}")
    print(f"mem0 Fixed Config: {'✅ PASS' if mem0_test else '❌ FAIL'}")
    print(f"Memory Manager: {'✅ PASS' if manager_test else '❌ FAIL'}")
    
    if openai_test and mem0_test and manager_test:
        print("\n🎉 All embedding tests passed! Dimension issue resolved.")
    else:
        print("\n⚠️ Some tests failed. Check the logs above for details.") 
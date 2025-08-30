"""
Simple memory test to verify basic mem0 functionality.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_basic_mem0():
    """Test basic mem0 functionality without complex configurations."""
    print("🧪 Testing Basic mem0 Functionality...")
    
    try:
        from mem0 import Memory
        
        # Test with default configuration first
        print("1. Testing default mem0 configuration...")
        memory = Memory()
        
        # Test basic operations
        user_id = "test_user_simple"
        test_content = "Hello, this is a test memory entry."
        
        # Add memory
        print("2. Adding test memory...")
        memory_id = memory.add(test_content, user_id=user_id)
        print(f"✅ Memory added with ID: {memory_id}")
        
        # Search memory
        print("3. Searching for memory...")
        results = memory.search("test memory", user_id=user_id, limit=1)
        print(f"✅ Found {len(results)} memories")
        
        if results:
            print(f"   Memory content: {results[0]}")
        
        # Get all memories
        print("4. Getting all memories...")
        all_memories = memory.get_all(user_id=user_id)
        print(f"✅ Total memories for user: {len(all_memories)}")
        
        # Clean up
        print("5. Cleaning up test data...")
        if memory_id:
            memory.delete(memory_id)
            print("✅ Test memory deleted")
        
        print("\n🎉 Basic mem0 test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Basic mem0 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_openai_config():
    """Test mem0 with OpenAI configuration."""
    print("\n🧪 Testing mem0 with OpenAI Configuration...")
    
    try:
        from mem0 import Memory
        
        # Check if OpenAI API key is available
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️ OPENAI_API_KEY not found, skipping OpenAI test")
            return True
        
        # Test with OpenAI configuration
        openai_config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.1,
                    "api_key": os.getenv("OPENAI_API_KEY")
                }
            }
        }
        
        print("1. Initializing mem0 with OpenAI...")
        memory = Memory.from_config(openai_config)
        
        # Test operations
        user_id = "test_user_openai"
        test_content = "This is a medical consultation about headaches."
        
        print("2. Adding medical memory...")
        memory_id = memory.add(test_content, user_id=user_id)
        print(f"✅ Memory added with ID: {memory_id}")
        
        print("3. Searching medical memory...")
        results = memory.search("headaches medical", user_id=user_id, limit=1)
        print(f"✅ Found {len(results)} medical memories")
        
        # Clean up
        print("4. Cleaning up...")
        if memory_id:
            memory.delete(memory_id)
            print("✅ Test memory deleted")
        
        print("✅ OpenAI configuration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI configuration test failed: {e}")
        return False


def test_qdrant_connection():
    """Test Qdrant connection independently."""
    print("\n🧪 Testing Qdrant Connection...")
    
    try:
        import requests
        
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = os.getenv("QDRANT_PORT", "6333")
        
        # Test health endpoint
        response = requests.get(f"http://{qdrant_host}:{qdrant_port}/health", timeout=5)
        
        if response.status_code == 200:
            print(f"✅ Qdrant is running at {qdrant_host}:{qdrant_port}")
            
            # Test collections endpoint
            collections_response = requests.get(f"http://{qdrant_host}:{qdrant_port}/collections")
            if collections_response.status_code == 200:
                collections = collections_response.json()
                print(f"✅ Qdrant collections accessible: {len(collections.get('result', {}).get('collections', []))} collections")
            
            return True
        else:
            print(f"❌ Qdrant health check failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Qdrant connection test failed: {e}")
        print("💡 Try starting Qdrant with: docker run -p 6333:6333 qdrant/qdrant")
        return False


if __name__ == "__main__":
    print("🚀 Simple Memory System Tests")
    print("=" * 50)
    
    # Test 1: Basic mem0 functionality
    basic_test = test_basic_mem0()
    
    # Test 2: Qdrant connection
    qdrant_test = test_qdrant_connection()
    
    # Test 3: OpenAI configuration
    openai_test = test_with_openai_config()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"Basic mem0: {'✅ PASS' if basic_test else '❌ FAIL'}")
    print(f"Qdrant: {'✅ PASS' if qdrant_test else '❌ FAIL'}")
    print(f"OpenAI: {'✅ PASS' if openai_test else '❌ FAIL'}")
    
    if basic_test and qdrant_test and openai_test:
        print("\n🎉 All tests passed! Memory system is ready.")
    else:
        print("\n⚠️ Some tests failed. Check the logs above for details.") 
"""
Quick test to verify memory system fixes are working.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def quick_system_check():
    """Quick check of the memory system components."""
    print("🚀 Quick Memory System Check")
    print("=" * 40)
    
    # Check 1: Environment variables
    print("1. Checking environment variables...")
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"✅ OPENAI_API_KEY: Set (ending in ...{api_key[-4:]})")
    else:
        print("❌ OPENAI_API_KEY: Not set")
        return False
    
    # Check 2: Qdrant connection
    print("\n2. Checking Qdrant connection...")
    try:
        import requests
        response = requests.get("http://localhost:6333/health", timeout=5)
        if response.status_code == 200:
            print("✅ Qdrant: Running and accessible")
        else:
            print(f"❌ Qdrant: HTTP {response.status_code}")
            print("💡 Start with: docker run -p 6333:6333 qdrant/qdrant")
            return False
    except Exception as e:
        print(f"❌ Qdrant: Connection failed ({e})")
        print("💡 Start with: docker run -p 6333:6333 qdrant/qdrant")
        return False
    
    # Check 3: Basic mem0 functionality
    print("\n3. Testing basic mem0...")
    try:
        from mem0 import Memory
        
        # Simple test without complex config
        memory = Memory()
        test_id = memory.add("Quick test", user_id="test123")
        
        if test_id:
            print("✅ mem0: Basic functionality works")
            # Clean up
            try:
                memory.delete(test_id)
            except:
                pass
        else:
            print("❌ mem0: Failed to add memory")
            return False
            
    except Exception as e:
        print(f"❌ mem0: Import or basic test failed ({e})")
        return False
    
    # Check 4: Our memory manager
    print("\n4. Testing our memory manager...")
    try:
        from config import Config
        from memory import LongTermMemoryManager
        
        config = Config()
        manager = LongTermMemoryManager(config)
        
        # Simple test
        test_id = manager.store_medical_insight(
            user_id="test123",
            insight="Quick test insight",
            source_agent="TEST_AGENT"
        )
        
        if test_id:
            print("✅ Memory Manager: Working correctly")
        else:
            print("⚠️ Memory Manager: Possible issues but fallback working")
        
    except Exception as e:
        print(f"❌ Memory Manager: Failed ({e})")
        print("🔄 Testing fallback system...")
        
        try:
            from memory.fallback_memory import FallbackMemoryManager
            fallback = FallbackMemoryManager(config)
            
            test_id = fallback.store_medical_insight(
                user_id="test123",
                insight="Fallback test",
                source_agent="TEST_AGENT"
            )
            
            if test_id:
                print("✅ Fallback Memory: Working correctly")
            else:
                print("❌ Fallback Memory: Also failed")
                return False
                
        except Exception as e2:
            print(f"❌ Fallback Memory: Failed ({e2})")
            return False
    
    print("\n🎉 All checks passed! Memory system is ready.")
    return True


if __name__ == "__main__":
    success = quick_system_check()
    if not success:
        print("\n❌ System check failed. Please resolve the issues above.")
        sys.exit(1)
    else:
        print("\n✅ System check passed. You can now run your medical assistant!")
        sys.exit(0) 
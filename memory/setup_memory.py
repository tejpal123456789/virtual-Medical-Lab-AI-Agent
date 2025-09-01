#!/usr/bin/env python3
"""
Setup script for long-term memory system.

This script helps initialize and configure the mem0-based memory system
for the Multi-Agent Medical Assistant.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = ["mem0ai", "redis", "qdrant-client"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "mem0ai==0.1.32", "redis==5.2.1"
            ])
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    return True


def check_environment_variables():
    """Check if required environment variables are set."""
    print("\n🔍 Checking environment variables...")
    
    required_vars = ["OPENAI_API_KEY"]
    optional_vars = ["QDRANT_HOST", "QDRANT_PORT", "NEO4J_URL"]
    
    missing_required = []
    
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var} is set")
        else:
            missing_required.append(var)
            print(f"❌ {var} is missing")
    
    for var in optional_vars:
        if os.getenv(var):
            print(f"✅ {var} is set")
        else:
            print(f"⚠️ {var} not set (using defaults)")
    
    if missing_required:
        print(f"\n❌ Missing required environment variables: {', '.join(missing_required)}")
        print("Please set these in your .env file before continuing.")
        return False
    
    return True


def check_services():
    """Check if required services are running."""
    print("\n🔍 Checking required services...")
    
    # Check Qdrant
    try:
        import requests
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = os.getenv("QDRANT_PORT", "6333")
        
        response = requests.get(f"http://{qdrant_host}:{qdrant_port}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Qdrant is running and healthy")
        else:
            print(f"⚠️ Qdrant responded with status {response.status_code}")
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        print("Please start Qdrant with: docker run -p 6333:6333 qdrant/qdrant")
        return False
    
    # Check Redis (optional, for enhanced features)
    try:
        import redis
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        redis_client = redis.from_url(redis_url)
        redis_client.ping()
        print("✅ Redis is running and accessible")
    except Exception as e:
        print(f"⚠️ Redis connection failed: {e} (optional service)")
    
    # Check Neo4j (optional)
    neo4j_url = os.getenv("NEO4J_URL")
    if neo4j_url:
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                neo4j_url,
                auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
            )
            driver.verify_connectivity()
            print("✅ Neo4j is running and accessible")
            driver.close()
        except Exception as e:
            print(f"⚠️ Neo4j connection failed: {e} (optional service)")
    else:
        print("⚠️ Neo4j not configured (optional service)")
    
    return True


def initialize_memory_system():
    """Initialize the memory system."""
    print("\n🚀 Initializing memory system...")
    
    try:
        from config import Config
        from memory import LongTermMemoryManager
        
        config = Config()
        memory_manager = LongTermMemoryManager(config)
        
        # Test basic operations
        test_user = "setup_test_user"
        
        # Store a test memory
        memory_id = memory_manager.memory.add(
            "System initialization test",
            user_id=test_user,
            metadata={"test": True, "setup_time": time.time()}
        )
        
        # Retrieve it
        memories = memory_manager.memory.search(
            query="initialization test",
            user_id=test_user,
            limit=1
        )
        
        # Clean up
        if memory_id:
            memory_manager.memory.delete(memory_id)
        
        if memories:
            print("✅ Memory system initialization successful")
            return True
        else:
            print("❌ Memory system test failed")
            return False
            
    except Exception as e:
        print(f"❌ Memory system initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_backup_directory():
    """Create backup directory for memory exports."""
    backup_dir = Path(__file__).parent / "backups"
    backup_dir.mkdir(exist_ok=True)
    print(f"✅ Backup directory created: {backup_dir}")


def main():
    """Main setup function."""
    print("🏥 Multi-Agent Medical Assistant - Memory System Setup")
    print("=" * 60)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed. Please resolve issues and try again.")
        return False
    
    # Step 2: Check environment variables
    if not check_environment_variables():
        print("❌ Environment check failed. Please configure required variables.")
        return False
    
    # Step 3: Check services
    if not check_services():
        print("❌ Service check failed. Please start required services.")
        return False
    
    # Step 4: Create backup directory
    create_backup_directory()
    
    # Step 5: Initialize memory system
    if not initialize_memory_system():
        print("❌ Memory system initialization failed.")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 Memory system setup completed successfully!")
    print("\nNext steps:")
    print("1. Run: python memory/test_memory_system.py")
    print("2. Start your medical assistant application")
    print("3. The memory system will automatically enhance all agent interactions")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
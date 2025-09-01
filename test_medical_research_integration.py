#!/usr/bin/env python3
"""
Test Medical Research Agent Integration
Tests the medical research functionality integrated into agent_decision.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.agent_decision import process_query
from langchain_core.messages import HumanMessage, AIMessage

def test_medical_research_routing():
    """Test that medical research queries are properly routed to MEDICAL_RESEARCH_AGENT"""
    
    print("🧪 Testing Medical Research Agent Integration")
    print("="*60)
    
    # Test queries that should route to medical research agent
    test_queries = [
        "Research the latest clinical trials for Alzheimer's disease treatment",
        "Find recent medical literature on immunotherapy for cancer",
        "What are the latest treatment guidelines for Type 2 diabetes?",
        "Conduct a literature review on COVID-19 long term effects",
        "Research current clinical trials for depression treatment"
    ]
    
    print("🔍 Environment Check:")
    required_vars = ["OPENAI_API_KEY", "BRIGHT_DATA_API_TOKEN"]
    for var in required_vars:
        status = "✅" if os.getenv(var) else "❌" 
        print(f"- {var}: {status}")
    
    if not all(os.getenv(var) for var in required_vars):
        print("⚠️ Missing required environment variables. Skipping tests.")
        return
    
    print(f"\n🧪 Running {len(test_queries)} test queries...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i}: Medical Research Query ---")
        print(f"Query: {query}")
        
        try:
            # Process the query through agent_decision system
            result = process_query(query)
            
            # Check if it was routed to MEDICAL_RESEARCH_AGENT
            agent_used = result.get("agent_name", "Unknown")
            print(f"🎯 Routed to: {agent_used}")
            
            if "MEDICAL_RESEARCH_AGENT" in agent_used:
                print("✅ Correctly routed to Medical Research Agent")
                
                # Check response quality
                output = result.get("output", "")
                if isinstance(output, AIMessage):
                    response_text = output.content
                elif isinstance(output, str):
                    response_text = output
                else:
                    response_text = str(output)
                
                print(f"📄 Response length: {len(response_text)} characters")
                print(f"📄 Response preview: {response_text[:200]}...")
                
                # Check for medical disclaimer
                if "Medical Disclaimer" in response_text:
                    print("✅ Medical disclaimer included")
                else:
                    print("⚠️ Medical disclaimer missing")
                    
            else:
                print(f"❌ Incorrectly routed to: {agent_used}")
                
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
    
    print(f"\n🏁 Medical Research Agent Integration Test Complete")

def test_simple_medical_research():
    """Simple test with one medical research query"""
    
    print("🧪 Simple Medical Research Test")
    print("="*40)
    
    query = "Research the latest treatment options for migraine headaches"
    print(f"Query: {query}")
    
    try:
        result = process_query(query)
        agent_used = result.get("agent_name", "Unknown")
        
        print(f"🎯 Agent: {agent_used}")
        
        if "MEDICAL_RESEARCH_AGENT" in agent_used:
            print("✅ Successfully routed to Medical Research Agent")
            
            output = result.get("output", "")
            if isinstance(output, AIMessage):
                response_text = output.content
            else:
                response_text = str(output)
                
            print(f"📄 Response ({len(response_text)} chars):")
            print("-" * 40)
            print(response_text)
            print("-" * 40)
            
        else:
            print(f"❌ Routed to wrong agent: {agent_used}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    print("🏥 Medical Research Agent Integration Tester")
    print("Choose test:")
    print("1. Simple test")
    print("2. Comprehensive routing test")
    
    choice = input("Enter choice (1-2): ").strip()
    
    if choice == "1":
        test_simple_medical_research()
    elif choice == "2":
        test_medical_research_routing()
    else:
        print("❌ Invalid choice. Running simple test by default.")
        test_simple_medical_research() 
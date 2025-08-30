#!/usr/bin/env python3
"""
Example: Using Medical Research Agent through Agent Decision System
Shows how the integrated MEDICAL_RESEARCH_AGENT works in the LangGraph flow
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.agent_decision import process_query
from langchain_core.messages import HumanMessage, AIMessage

def example_diabetes_research():
    """Example: Research Type 2 Diabetes treatments"""
    
    print("🏥 Example: Diabetes Research through Agent Decision System")
    print("="*60)
    
    query = "Research the latest evidence-based treatment guidelines for Type 2 Diabetes from major medical organizations and recent clinical trials"
    
    print(f"📝 Query: {query}")
    print("\n🔄 Processing through agent decision system...")
    
    try:
        result = process_query(query)
        
        print(f"\n🎯 Agent Used: {result.get('agent_name', 'Unknown')}")
        
        # Display the research results
        output = result.get("output", "")
        if isinstance(output, AIMessage):
            response_text = output.content
        else:
            response_text = str(output)
            
        print(f"\n📊 Research Results ({len(response_text)} characters):")
        print("="*60)
        print(response_text)
        print("="*60)
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def example_cancer_immunotherapy():
    """Example: Research Cancer Immunotherapy"""
    
    print("🏥 Example: Cancer Immunotherapy Research")
    print("="*50)
    
    query = "Find recent clinical trials and medical literature on CAR-T cell therapy for treating blood cancers"
    
    print(f"📝 Query: {query}")
    
    try:
        result = process_query(query)
        
        print(f"🎯 Agent: {result.get('agent_name', 'Unknown')}")
        
        output = result.get("output", "")
        if isinstance(output, AIMessage):
            response_text = output.content
        else:
            response_text = str(output)
            
        print(f"\n📊 Research Results:")
        print("-" * 50)
        print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
        print("-" * 50)
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def interactive_medical_research():
    """Interactive medical research session"""
    
    print("🏥 Interactive Medical Research Session")
    print("="*50)
    print("Enter medical research topics or 'quit' to exit")
    
    while True:
        query = input("\n🔬 Research Topic: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Session ended")
            break
            
        if not query:
            print("⚠️ Please enter a research topic")
            continue
            
        try:
            print(f"\n🔄 Researching: {query}")
            result = process_query(query)
            
            agent_used = result.get("agent_name", "Unknown")
            print(f"🎯 Processed by: {agent_used}")
            
            # Show results
            output = result.get("output", "")
            if isinstance(output, AIMessage):
                response_text = output.content
            else:
                response_text = str(output)
                
            print(f"\n📊 Results:")
            print("-" * 40)
            print(response_text)
            print("-" * 40)
            
        except Exception as e:
            print(f"❌ Research failed: {e}")

def main():
    """Main function to choose example type"""
    
    print("🏥 Medical Research Agent Examples")
    print("Choose an example:")
    print("1. Diabetes research example")
    print("2. Cancer immunotherapy example") 
    print("3. Interactive research session")
    print("4. Check environment setup")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        example_diabetes_research()
    elif choice == "2":
        example_cancer_immunotherapy()
    elif choice == "3":
        interactive_medical_research()
    elif choice == "4":
        # Environment check
        print("\n🔍 Environment Check:")
        required_vars = ["OPENAI_API_KEY", "BRIGHT_DATA_API_TOKEN"]
        for var in required_vars:
            status = "✅" if os.getenv(var) else "❌"
            print(f"- {var}: {status}")
        
        if all(os.getenv(var) for var in required_vars):
            print("✅ Environment setup complete!")
        else:
            print("❌ Missing environment variables. Check your .env file.")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main() 
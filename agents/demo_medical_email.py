#!/usr/bin/env python3
"""
Demo: Medical Research Email Assistant
Shows how to research medical topics and send results via Gmail
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from medical_research_emailer import get_medical_research, send_research_email

async def demo_diabetes_research():
    """Demo: Research diabetes and send via email"""
    
    print("🧪 DEMO: Diabetes Research Email")
    print("="*50)
    
    # 1. Conduct research
    research_topic = "Type 2 Diabetes latest treatment guidelines 2024"
    print(f"🔬 Step 1: Researching {research_topic}")
    
    research_content = await get_medical_research(research_topic)
    
    # 2. Display research
    print(f"\n📄 Step 2: Research Results ({len(research_content)} characters)")
    print("-" * 40)
    print(research_content)
    print("-" * 40)
    
    # 3. Send email (you can change this email)
    recipient_email = "your-email@example.com"  # ⚠️ CHANGE THIS TO YOUR EMAIL
    
    print(f"\n📧 Step 3: Sending to {recipient_email}")
    
    if input("Continue with email sending? [y/n]: ").lower() == 'y':
        success = send_research_email(research_content, recipient_email, research_topic)
        
        if success:
            print("✅ Demo completed successfully!")
        else:
            print("❌ Demo failed during email sending")
    else:
        print("📧 Email sending skipped")

async def demo_cancer_research():
    """Demo: Cancer research and email"""
    
    print("🧪 DEMO: Cancer Research Email")
    print("="*50)
    
    research_topic = "Immunotherapy cancer treatment breakthroughs 2024"
    recipient_email = "medical-team@example.com"  # ⚠️ CHANGE THIS
    
    print(f"🔬 Researching: {research_topic}")
    research_content = await get_medical_research(research_topic)
    
    print(f"📧 Sending to: {recipient_email}")
    success = send_research_email(research_content, recipient_email, research_topic)
    
    return success

async def main():
    """Choose demo to run"""
    
    print("🏥 Medical Research Email Demo")
    print("Choose a demo:")
    print("1. Diabetes research")
    print("2. Cancer research") 
    print("3. Custom research")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        await demo_diabetes_research()
    elif choice == "2":
        await demo_cancer_research()
    elif choice == "3":
        # Custom research
        topic = input("Enter research topic: ").strip()
        email = input("Enter recipient email: ").strip()
        
        if topic and email:
            content = await get_medical_research(topic)
            success = send_research_email(content, email, topic)
            print("✅ Custom demo completed!" if success else "❌ Custom demo failed")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    print("⚠️  Before running:")
    print("1. Make sure your .env file has all required API keys")
    print("2. Update recipient email addresses in the demo functions")
    print("3. Ensure you have authorized Gmail access in Arcade")
    print()
    
    asyncio.run(main()) 
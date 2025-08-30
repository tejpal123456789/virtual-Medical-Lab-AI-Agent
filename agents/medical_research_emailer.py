#!/usr/bin/env python3
"""
Simple Medical Research Email Sender
Combines bright_data research with Gmail_SendEmail from Arcade
"""

import asyncio
import os
import uuid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from mcp_use.client import MCPClient
from mcp_use.adapters.langchain_adapter import LangChainAdapter
from langchain_arcade import ToolManager
from arcadepy import Arcade
import datetime

# Load environment variables
load_dotenv()

# Create checkpointer
checkpointer = MemorySaver()

async def setup_research_tools():
    """Setup Bright Data tools for research"""
    bright_data_config = {
        "mcpServers": {
            "Bright Data": {
                "command": "npx", 
                "args": ["@brightdata/mcp"],
                "env": {
                    "API_TOKEN": os.getenv("BRIGHT_DATA_API_TOKEN"),
                }
            }
        }
    }
    
    client = MCPClient.from_dict(bright_data_config)
    adapter = LangChainAdapter()
    tools = await adapter.create_tools(client)
    print(f"✅ Research tools ready: {len(tools)}")
    return tools

def setup_gmail_tools():
    """Setup Arcade Gmail tools - following tool_arcade.py pattern exactly"""
    
    # Initialize Arcade client
    arcade_client = Arcade(api_key=os.getenv("ARCADE_API_KEY"))
    manager = ToolManager(client=arcade_client)
    
    # Get Gmail tools
    gmail_tools = manager.init_tools(tools=["Gmail_SendEmail"])
    
    # Authorize Gmail_SendEmail tool
    def authorize_tool(tool_name, user_id, manager):
        auth_response = manager.authorize(tool_name=tool_name, user_id=user_id)
        if auth_response.status != "completed":
            print(f"Please authorize {tool_name}: {auth_response.url}")
            manager.wait_for_auth(auth_response.id)
    
    for tool in gmail_tools:
        authorize_tool(tool.name, os.getenv("ARCADE_USER_ID"), manager)
    
    return gmail_tools

async def get_medical_research(research_topic):
    """Get medical research using Bright Data tools"""
    
    # Setup research tools
    research_tools = await setup_research_tools()
    
    # Create research agent
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.1,
        max_tokens=2000
    )
    
    current_date = datetime.datetime.now().strftime("%B %d, %Y")
    
    research_agent = create_react_agent(
        model=llm,
        tools=research_tools,
        prompt=f"""You are a medical research assistant. Today is {current_date}.

Use your web scraping tools to find current medical information. 
IMPORTANT: Keep your final summary under 1200 characters for email compatibility.

Research Process:
1. Search for recent medical studies and news
2. Extract key findings and statistics
3. Provide concise, actionable medical insights
4. Focus on evidence-based information only

Always cite sources when possible."""
    )
    
    # Conduct research
    print(f"🔬 Researching: {research_topic}")
    
    research_result = await research_agent.ainvoke({
        "messages": [("human", f"Research the latest medical information about: {research_topic}. "
                              f"Provide a concise summary with key findings, statistics, and recommendations. "
                              f"Keep it under 1200 characters total.")]
    })
    
    research_content = research_result["messages"][-1].content
    print(f"✅ Research complete ({len(research_content)} chars)")
    
    return research_content

def send_research_email(research_content, recipient_email, research_topic):
    """Send research content via Gmail using Arcade - following tool_arcade.py pattern"""
    
    # Setup Gmail tools
    gmail_tools = setup_gmail_tools()
    
    # Create Gmail agent - exact pattern from tool_arcade.py
    gmail_agent = create_react_agent(
        model="openai:gpt-5",
        prompt="You are a helpful assistant that can help with everyday tasks."
               " If the user's request is confusing you must ask them to clarify"
               " their intent, and fulfill the instruction to the best of your"
               " ability. Be concise and friendly at all times."
               " Use the Gmail tools to address requests about reading or sending emails.",
        tools=gmail_tools,
        checkpointer=checkpointer
    )
    
    # Config with user_id - exact pattern from tool_arcade.py
    config = {
        "configurable": {
            "thread_id": uuid.uuid4(),
            "user_id": os.getenv("ARCADE_USER_ID")
        }
    }
    
    print(f"📧 Sending to: {recipient_email}")
    print(f"🔍 Using user_id: {os.getenv('ARCADE_USER_ID')}")
    
    # Create email prompt
    email_prompt = f"""Send an email with:
    
Subject: Medical Research Summary - {research_topic}
Recipient: {recipient_email}
Body: 
Dear Colleague,

Here is the latest medical research summary on {research_topic}:

{research_content}

Best regards,
Medical Research Assistant
    """
    
    user_message = {"messages": [HumanMessage(content=email_prompt)]}
    
    # Send email using the exact streaming pattern from tool_arcade.py
    try:
        for event in gmail_agent.stream(user_message, config=config, stream_mode="values"):
            if "messages" in event:
                event["messages"][-1].pretty_print()
        return True
    except Exception as e:
        print(f"❌ Email sending failed: {e}")
        return False

async def main():
    """Main function - simple workflow"""
    
    print("🏥 Medical Research Email Assistant")
    print("="*50)
    
    # Check environment
    required_vars = ["OPENAI_API_KEY", "BRIGHT_DATA_API_TOKEN", "ARCADE_API_KEY", "ARCADE_USER_ID"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        return
    
    print("✅ All environment variables present")
    
    # Get research topic
    research_topic = input("Enter medical research topic: ").strip()
    if not research_topic:
        print("❌ No research topic provided")
        return
    
    # Get research content
    research_content = await get_medical_research(research_topic)
    
    # Show preview
    print(f"\n📋 Research Preview:")
    print("-" * 40) 
    print(research_content[:400] + "..." if len(research_content) > 400 else research_content)
    print("-" * 40)
    
    # Get recipient email
    recipient = input("Enter recipient email: ").strip()
    if not recipient:
        print("❌ No recipient provided")
        return
    
    # Send email
    print(f"\n📤 Sending research to {recipient}...")
    success = send_research_email(research_content, recipient, research_topic)
    
    if success:
        print("✅ Medical research email sent successfully!")
    else:
        print("❌ Failed to send email")

# Quick test function
async def quick_test():
    """Quick test with predefined values"""
    research_topic = "COVID-19 latest treatments"
    recipient = "test@example.com"  # Change this to your email
    
    print(f"🧪 Quick test: Researching {research_topic}")
    research_content = await get_medical_research(research_topic)
    
    print(f"📧 Test sending to {recipient}")
    success = send_research_email(research_content, recipient, research_topic)
    
    return success

if __name__ == "__main__":
    # Uncomment one of these:
    
    # For interactive mode:
    asyncio.run(main())
    
    # For quick test mode:
    # asyncio.run(quick_test()) 
#!/usr/bin/env python3
"""
Combined Medical Research Assistant with Email Delivery
This script combines the web research capabilities from bright_data.py 
with the Gmail sending functionality from tool_arcade.py
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
from typing import Callable, Any
from langchain_core.tools import tool, BaseTool
from langchain_core.runnables import RunnableConfig
import pprint
import threading
import time

# Load environment variables
load_dotenv()

# Create checkpointer for state persistence
checkpointer = MemorySaver()

async def setup_bright_data_tools():
    """Configure Bright Data MCP client and create LangChain-compatible tools"""
    
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
    
    print(f"✅ Connected to Bright Data MCP server")
    print(f"📊 Available Bright Data tools: {len(tools)}")
    
    return tools

def setup_arcade_gmail_tools():
    """Setup Arcade Gmail tools with authentication"""
    
    # Initialize Arcade client
    arcade_client = Arcade(api_key=os.getenv("ARCADE_API_KEY"))
    manager = ToolManager(client=arcade_client)
    
    # Initialize Gmail tools
    gmail_tools = manager.init_tools(tools=["Gmail_ListEmails", "Gmail_SendEmail"])
    
    # Authorize tools
    for tool in gmail_tools:
        authorize_tool(tool.name, os.getenv("ARCADE_USER_ID"), manager)
    
    print(f"✅ Gmail tools authorized and ready")
    return gmail_tools, manager

def authorize_tool(tool_name, user_id, manager):
    """Authorize a specific tool for the user"""
    auth_response = manager.authorize(
        tool_name=tool_name,
        user_id=user_id
    )
    if auth_response.status != "completed":
        print(f"The app wants to use the {tool_name} tool.\n"
              f"Please click this url to authorize it {auth_response.url}")
        manager.wait_for_auth(auth_response.id)

def add_human_in_the_loop(target_tool: Callable | BaseTool) -> BaseTool:
    """Wrap a tool to support human-in-the-loop review."""
    if not isinstance(target_tool, BaseTool):
        target_tool = tool(target_tool)

    @tool(
        target_tool.name,
        description=target_tool.description,
        args_schema=target_tool.args_schema
    )
    def call_tool_with_interrupt(config: RunnableConfig = None, **tool_input):
        # Format arguments for display
        arguments = pprint.pformat(tool_input, indent=4)
        
        # Simple console-based approval
        print(f"\n🔍 Tool Request: {target_tool.name}")
        print(f"Arguments:\n{arguments}")
        user_approval = input("Do you approve this action? [y/n]: ")
        
        if user_approval.lower() == "y":
            print(f"✅ Executing {target_tool.name}...")
            try:
                result = {"response": None, "error": None, "completed": False}
                
                def execute_tool():
                    try:
                        print(f"🔄 Calling {target_tool.name} with Arcade...")
                        
                        if config and hasattr(config, 'get') and config.get('configurable', {}).get('user_id'):
                            print(f"🔍 Using user_id: {config['configurable']['user_id']}")
                            response = target_tool.invoke(tool_input, config)
                        elif config:
                            print(f"🔍 Config exists but no user_id found, trying anyway...")
                            response = target_tool.invoke(tool_input, config)
                        else:
                            print(f"🔍 No config provided, calling without config...")
                            response = target_tool.invoke(tool_input)
                            
                        result["response"] = response
                        result["completed"] = True
                        print(f"🔄 Tool returned: {type(response)}")
                    except Exception as e:
                        result["error"] = str(e)
                        result["completed"] = True
                
                # Start tool execution in a separate thread
                thread = threading.Thread(target=execute_tool)
                thread.daemon = True
                thread.start()
                
                # Wait for completion with timeout
                timeout_seconds = 30
                start_time = time.time()
                
                while not result["completed"] and (time.time() - start_time) < timeout_seconds:
                    time.sleep(0.5)
                    print(".", end="", flush=True)
                
                print()  # New line after dots
                
                if not result["completed"]:
                    print(f"⏰ Tool {target_tool.name} timed out after {timeout_seconds} seconds")
                    return f"Tool execution timed out after {timeout_seconds} seconds"
                
                if result["error"]:
                    print(f"❌ Tool {target_tool.name} failed: {result['error']}")
                    return f"Tool execution failed: {result['error']}"
                
                tool_response = result["response"]
                print(f"✅ Tool {target_tool.name} completed successfully")
                print(f"📄 Response preview: {str(tool_response)[:200]}...")
                return tool_response
                
            except Exception as e:
                print(f"❌ Unexpected error in {target_tool.name}: {str(e)}")
                return f"Tool execution failed: {str(e)}"
        else:
            print(f"❌ Tool {target_tool.name} was denied by user")
            return "The user did not approve this action."

    return call_tool_with_interrupt

async def create_research_agent(bright_data_tools):
    """Create a medical research agent using Bright Data tools"""
    
    current_date = datetime.datetime.now().strftime("%B %d, %Y")
    
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-4o-mini",
        temperature=0.1,
        max_tokens=4000,
        model_kwargs={
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
    )
    
    system_prompt = f"""You are a medical research assistant. Today's date is {current_date}.

You have specialized tools for web scraping and data extraction. When conducting medical research, you MUST use these tools to get current, up-to-date information.

CRITICAL: Always summarize and condense the research data. Focus on key medical findings, statistics, and insights only.

Medical Research Process:
1. Search for recent medical studies, news, and research papers
2. Extract key findings and statistics  
3. Analyze trends and developments
4. SUMMARIZE results concisely (max 1500 characters for email compatibility)
5. Focus on actionable medical insights

Always use tools for current medical data requests. Keep responses focused and evidence-based."""
    
    agent = create_react_agent(
        model=llm,
        tools=bright_data_tools,
        prompt=system_prompt
    )
    
    return agent

def create_gmail_agent(gmail_tools):
    """Create an agent specifically for Gmail operations"""
    
    # Protect Gmail tools with human-in-the-loop
    tools_to_protect = ["Gmail_SendEmail"]
    protected_gmail_tools = [
        add_human_in_the_loop(t) if t.name in tools_to_protect else t
        for t in gmail_tools
    ]
    
    agent = create_react_agent(
        model="openai:gpt-5",
        prompt="You are a Gmail assistant that helps send emails with research content. "
               "Be precise with email formatting and always confirm recipients before sending. "
               "Format the email content clearly with proper subject lines.",
        tools=protected_gmail_tools,
        checkpointer=checkpointer
    )
    
    return agent

async def conduct_medical_research(research_agent, research_topic):
    """Conduct medical research on a specific topic"""
    
    print(f"🔬 Starting medical research on: {research_topic}")
    print("="*60)
    
    research_prompt = f"""
    Please conduct comprehensive medical research on: {research_topic}
    
    I need you to:
    1. Search for the latest medical studies and clinical trials
    2. Find recent news and developments in this area
    3. Look up expert opinions and medical guidelines
    4. Summarize key findings, statistics, and treatment recommendations
    
    IMPORTANT: Keep the summary concise and under 1500 characters total, suitable for email delivery.
    Focus on the most important medical insights and actionable information.
    """
    
    research_result = await research_agent.ainvoke({
        "messages": [("human", research_prompt)]
    })
    
    research_content = research_result["messages"][-1].content
    print("\n📋 Research completed!")
    print(f"📄 Content length: {len(research_content)} characters")
    
    return research_content

async def send_research_email(gmail_agent, config, research_content, recipient_email, research_topic):
    """Send the research content via email using Gmail agent"""
    
    print(f"📧 Preparing to send research email to: {recipient_email}")
    
    email_prompt = f"""
    Please send an email with the following details:
    
    Recipient: {recipient_email}
    Subject: Medical Research Summary: {research_topic}
    
    Email Body:
    {research_content}
    
    Please format this as a professional medical research summary email.
    """
    
    user_message = {"messages": [HumanMessage(content=email_prompt)]}
    
    print("📤 Sending email through Gmail agent...")
    
    try:
        for event in gmail_agent.stream(user_message, config=config, stream_mode="values"):
            if "messages" in event:
                event["messages"][-1].pretty_print()
                
    except Exception as e:
        print(f"❌ Error sending email: {e}")
        return False
        
    return True

async def main():
    """Main function that orchestrates research and email sending"""
    
    print("🚀 Starting Combined Medical Research & Email Assistant")
    print("="*60)
    
    # Setup tools
    print("🔧 Setting up tools...")
    bright_data_tools = await setup_bright_data_tools()
    gmail_tools, manager = setup_arcade_gmail_tools()
    
    # Create agents
    print("🤖 Creating agents...")
    research_agent = await create_research_agent(bright_data_tools)
    gmail_agent = create_gmail_agent(gmail_tools)
    
    # Configuration for Gmail agent
    config = {
        "configurable": {
            "thread_id": uuid.uuid4(),
            "user_id": os.getenv("ARCADE_USER_ID")
        }
    }
    
    print(f"🔍 Thread ID: {config['configurable']['thread_id']}")
    print(f"🔍 User ID: {os.getenv('ARCADE_USER_ID')}")
    
    # Interactive loop for research and email sending
    while True:
        print("\n" + "="*60)
        print("Medical Research Assistant")
        print("1. Conduct medical research")
        print("2. Send research via email") 
        print("3. Exit")
        
        choice = input("Choose an option (1-3): ").strip()
        
        if choice == "1":
            research_topic = input("Enter medical research topic: ").strip()
            if research_topic:
                research_content = await conduct_medical_research(research_agent, research_topic)
                
                # Store the research content for email sending
                globals()['last_research_content'] = research_content
                globals()['last_research_topic'] = research_topic
                
                print(f"\n📋 Research Summary Preview:")
                print("-" * 40)
                print(research_content[:300] + "..." if len(research_content) > 300 else research_content)
                print("-" * 40)
        
        elif choice == "2":
            if 'last_research_content' not in globals():
                print("❌ No research content available. Please conduct research first (option 1).")
                continue
                
            recipient = input("Enter recipient email address: ").strip()
            if recipient:
                success = await send_research_email(
                    gmail_agent, 
                    config, 
                    globals()['last_research_content'], 
                    recipient, 
                    globals()['last_research_topic']
                )
                
                if success:
                    print("✅ Research email sent successfully!")
                else:
                    print("❌ Failed to send research email.")
        
        elif choice == "3":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    print("🏥 Medical Research Assistant with Email Delivery")
    print("Required environment variables:")
    print(f"- OPENAI_API_KEY: {'✅' if os.getenv('OPENAI_API_KEY') else '❌'}")
    print(f"- BRIGHT_DATA_API_TOKEN: {'✅' if os.getenv('BRIGHT_DATA_API_TOKEN') else '❌'}")
    print(f"- ARCADE_API_KEY: {'✅' if os.getenv('ARCADE_API_KEY') else '❌'}")
    print(f"- ARCADE_USER_ID: {'✅' if os.getenv('ARCADE_USER_ID') else '❌'}")
    
    if all([os.getenv(key) for key in ["OPENAI_API_KEY", "BRIGHT_DATA_API_TOKEN", "ARCADE_API_KEY", "ARCADE_USER_ID"]]):
        asyncio.run(main())
    else:
        print("❌ Missing required environment variables. Please check your .env file.") 
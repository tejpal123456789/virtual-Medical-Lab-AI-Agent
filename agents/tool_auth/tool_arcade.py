from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
import uuid
from dotenv import load_dotenv
load_dotenv()

# create a checkpointer to persist the graph's state
checkpointer = MemorySaver()

agent_a = create_react_agent(
    model="openai:gpt-5",
    prompt="You are a helpful assistant that can help with everyday tasks."
           " If the user's request is confusing you must ask them to clarify"
           " their intent, and fulfill the instruction to the best of your"
           " ability. Be concise and friendly at all times.",
    tools=[], # no tools for now!
    checkpointer=checkpointer
)


from langgraph.graph.state import CompiledStateGraph
def run_graph(graph: CompiledStateGraph, config, input):
    for event in graph.stream(input, config=config, stream_mode="values"):
        if "messages" in event:
            event["messages"][-1].pretty_print()


# the configuration helps LangGraph keep track of conversations and interrups
# While it's not needed for this agent. The agent will remember different
# conversations based on the thread_id. This code generates a random id every
# time you run the cell, but you can hardcode the thread_id if you want to
# test the memory.
# config = {
#     "configurable": {
#         "thread_id": uuid.uuid4()
#     }
# }
# while True:
#     user_input = input("👤: ")
#     # let's use "exit" as a safe way to break the infinite loop
#     if user_input.lower() == "exit":
#         break

#     user_message = {"messages": [HumanMessage(content=user_input)]}
#     run_graph(agent_a, config, user_message)
# config = {
#     "configurable": {
#         "thread_id": uuid.uuid4()
#     }
# }
# print(f'thread_id = {config["configurable"]["thread_id"]}')

# prompt = "what's today's date?"
# user_message = {"messages": [HumanMessage(content=prompt)]}
# run_graph(agent_a, config, user_message)


# # authenticatiom
# config = {
#     "configurable": {
#         "thread_id": uuid.uuid4()
#     }
# }
# print(f'thread_id = {config["configurable"]["thread_id"]}')

# prompt = "summarize my latest 3 emails please"
# user_message = {"messages": [HumanMessage(content=prompt)]}
# run_graph(agent_a, config, user_message)


# tool authentication 

import os
from langchain_arcade import ToolManager
from arcadepy import Arcade

arcade_client = Arcade(api_key=os.getenv("ARCADE_API_KEY"))
print(arcade_client)
manager = ToolManager(client=arcade_client)
print(manager)
gmail_tool = manager.init_tools(tools=["Gmail_ListEmails"])[0]

def authorize_tool(tool_name, user_id, manager):
    # This line will check if this user is authorized to use the
    # tool, and return a response that we can use if the user
    # did not authorize the tool yet.
    auth_response = manager.authorize(
        tool_name=tool_name,
        user_id=user_id
    )
    if auth_response.status != "completed":
        print(f"The app wants to use the {tool_name} tool.\n"
              f"Please click this url to authorize it {auth_response.url}")
        # wait until the user authorizes
        manager.wait_for_auth(auth_response.id)

authorize_tool(gmail_tool.name, os.getenv("ARCADE_USER_ID"), manager)

# Test the tool directly before using in agent
print("\n🧪 Testing Gmail tool directly...")
try:
    # Create a test config with user_id
    test_config = {
        "configurable": {
            "user_id": os.getenv("ARCADE_USER_ID")
        }
    }
    test_result = gmail_tool.invoke({"n_emails": 1}, test_config)
    print(f"✅ Direct tool test successful: {type(test_result)}")
    print(f"📄 Sample result: {str(test_result)[:200]}...")
except Exception as e:
    print(f"❌ Direct tool test failed: {e}")
    print("This might indicate an authentication or API issue.")


# define a new agent, this time with access to our tool!
agent_b = create_react_agent(
    model="openai:gpt-5",
    prompt="You are a helpful assistant that can help with everyday tasks."
           " If the user's request is confusing you must ask them to clarify"
           " their intent, and fulfill the instruction to the best of your"
           " ability. Be concise and friendly at all times."
           # It's useful to let the agent know about the tools it has at its disposal.
           " Use the Gmail tools that you have to address requests about emails.",
    tools=[gmail_tool], # we pass the tool we previously authorized.
    checkpointer=checkpointer
)

config = {
    "configurable": {
        "thread_id": uuid.uuid4(),
        "user_id": os.getenv("ARCADE_USER_ID") # When using Arcade tools, we must provide the user_id on the LangGraph config, so Arcade can execute the tool invoked by the agent.
    }
}
print(f'thread_id = {config["configurable"]["thread_id"]}')

# we're using the same prompt we use before, but we're swapping the agent
prompt = "summarize my latest 3 emails please"
user_message = {"messages": [HumanMessage(content=prompt)]}
#run_graph(agent_b, config, user_message)

for tool_name, _ in manager:
    print(tool_name)
     

tools_to_protect = [
    "Gmail_ListEmails",
    "Gmail_SendEmail"
    # "Slack_SendDmToUser",
    # "Slack_SendMessage",
    # "Slack_SendMessageToChannel",
    # "NotionToolkit_AppendContentToEndOfPage",
    # "NotionToolkit_CreatePage",
]

from typing import Callable, Any
from langchain_core.tools import tool, BaseTool
from langgraph.types import interrupt, Command
from langchain_core.runnables import RunnableConfig
import pprint


def add_human_in_the_loop(
    target_tool: Callable | BaseTool,
) -> BaseTool:
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
        
        # Simple console-based approval (no LangGraph interrupt)
        print(f"\n🔍 Tool Request: {target_tool.name}")
        print(f"Arguments:\n{arguments}")
        user_approval = input("Do you approve this action? [y/n]: ")
        
        if user_approval.lower() == "y":
            print(f"✅ Executing {target_tool.name}...")
            try:
                # Execute with timeout using threading
                import threading
                import time
                
                result = {"response": None, "error": None, "completed": False}
                
                def execute_tool():
                    try:
                        print(f"🔄 Calling {target_tool.name} with Arcade...")
                        print(f"🔍 Config received: {config}")
                        print(f"🔍 Tool input: {tool_input}")
                        
                        # Try different ways to pass the config
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

protected_tools = [
    add_human_in_the_loop(t)
    if t.name in tools_to_protect else t
    for t in manager.to_langchain()
]
print(protected_tools)
def yes_no_loop(prompt: str) -> str:
    """
    Force the user to say yes or no
    """
    print(prompt)
    user_input = input("Your response [y/n]: ")
    while user_input.lower() not in ["y", "n"]:
        user_input = input("Your response (must be 'y' or 'n'): ")
    return "yes" if user_input.lower() == "y" else "no"


def handle_interrupts(graph: CompiledStateGraph, config):
    """Handle interrupts in the updated LangGraph format"""
    try:
        # Get current state
        state = graph.get_state(config)
        
        # Check if graph is interrupted (waiting for human input)
        if state.next:
            print(f"Graph is interrupted at: {state.next}")
            # The interrupt message should be in the last message or state
            if state.values.get("messages"):
                last_msg = state.values["messages"][-1]
                if hasattr(last_msg, 'content') and "Do you allow" in last_msg.content:
                    approved = yes_no_loop(last_msg.content)
                    # Resume with the approval
                    graph.update_state(config, {"approval": approved})
                    continue_execution(graph, config)
    except Exception as e:
        print(f"Error handling interrupts: {e}")
        print("Continuing without interrupt handling...")

def continue_execution(graph: CompiledStateGraph, config):
    """Continue graph execution after interrupt"""
    for event in graph.stream(None, config=config, stream_mode="values"):
        if "messages" in event:
            event["messages"][-1].pretty_print()

# define a new agent, this time with access to our tool!
agent_hitl = create_react_agent(
    model="openai:gpt-5",
    prompt="You are a helpful assistant that can help with everyday tasks."
           " If the user's request is confusing you must ask them to clarify"
           " their intent, and fulfill the instruction to the best of your"
           " ability. Be concise and friendly at all times."
           # It's useful to let the agent know about the tools it has at its disposal.
           " Use the Gmail tools to address requests about reading or sending emails."
           " Use the Slack tools to address requests about interactions with users and channels in Slack."
           " Use the Notion tools to address requests about managing content in Notion Pages."
           " In general, when possible, use the most relevant tool for the job.",
    tools=protected_tools,
    checkpointer=checkpointer
)
config = {
    "configurable": {
        "thread_id": uuid.uuid4(),
        "user_id": os.getenv("ARCADE_USER_ID") # When using Arcade tools, we must provide the user_id on the LangGraph config, so Arcade can execute the tool invoked by the agent.
    }
}
print(f'thread_id = {config["configurable"]["thread_id"]}')

# we're using the same prompt we use before, but we're swapping the agent
prompt = 'send an email with subject "confidential data" and body "this is top secret information" to thetajas@gmail.com'
user_message = {"messages": [HumanMessage(content=prompt)]}
#run_graph(agent_hitl, config, user_message)


config = {
    "configurable": {
        "thread_id": uuid.uuid4(),
        "user_id": os.getenv("ARCADE_USER_ID")  # Add user_id to main config
    }
}
print(f"🔍 ARCADE_USER_ID from env: {os.getenv('ARCADE_USER_ID')}")
print(f"🔍 Main config being used: {config}")

while True:
    user_input = input("👤: ")
    # let's use "exit" as a safe way to break the infinite loop
    if user_input.lower() == "exit":
        break

    user_message = {"messages": [HumanMessage(content=user_input)]}

    # Simple streaming without interrupt handling (approval happens in tool wrapper)
    try:
        for event in agent_hitl.stream(user_message, config=config, stream_mode="values"):
            if "messages" in event:
                event["messages"][-1].pretty_print()
                
    except Exception as e:
        print(f"Error: {e}")
        # Reset thread on error
        config = {
            "configurable": {
                "thread_id": uuid.uuid4(),
                "user_id": os.getenv("ARCADE_USER_ID")
            }
        }
        print(f"🔄 New thread_id = {config['configurable']['thread_id']}")
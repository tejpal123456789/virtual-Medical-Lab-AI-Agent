# Import all necessary libraries
import asyncio
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from mcp_use.client import MCPClient
from mcp_use.adapters.langchain_adapter import LangChainAdapter
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Verify environment setup
print("✅ Environment setup complete!")
print(f"Openai API Key loaded: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")

async def setup_bright_data_tools():
    """
    Configure Bright Data MCP client and create LangChain-compatible tools
    """
    
    # Configure the MCP server connection
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
    
    # Create MCP client and adapter
    client = MCPClient.from_dict(bright_data_config)
    adapter = LangChainAdapter()
    
    # Convert MCP tools to LangChain-compatible format
    tools = await adapter.create_tools(client)
    
    print(f"✅ Connected to Bright Data MCP server")
    print(f"📊 Available tools: {len(tools)}")
    
    return tools

# Test the connection


import datetime

async def create_web_scraper_agent(tools):
    """
    Create a ReAct agent configured for intelligent web scraping
    """
    
    # Get the tools from Bright Data first
    #tools = await setup_bright_data_tools()
    
    # Get current date
    current_date = datetime.datetime.now().strftime("%B %d, %Y")
    
    # Initialize the language model with context management
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-4o-mini",  # Fast and capable model for reasoning
        temperature=0.1, # Low temperature for consistent, focused responses
        max_tokens=4000,  # Limit response length
        # Add token management
        model_kwargs={
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
    )
    
    # Define comprehensive system prompt for the agent with dynamic date and tool count
    system_prompt = f"""You are a web data extraction agent. Today's date is {current_date}.

You have {len(tools)} specialized tools for web scraping and data extraction. When users request web data or current information, you MUST use these tools - do not rely on your training data.

IMPORTANT: Always summarize and condense the extracted data to avoid context length issues. Focus on key information only.

Available capabilities:
- Search engines (Google/Bing/Yandex)
- Universal web scraping (any website)
- Platform extractors (Amazon, LinkedIn, Instagram, Facebook, X, TikTok, YouTube, Reddit, etc.)
- Browser automation

Process:
1. Identify data need
2. Select appropriate tool
3. Execute extraction
4. SUMMARIZE results (max 2000 characters)
5. Return structured, concise results

Always use tools for current/live data requests. Keep responses concise and focused."""
    
    # Create the ReAct agent
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt
    )
    
    print("🤖 ReAct Web Scraper Agent created successfully!")
    return agent

async def test_basic_search(agent):
    """
    Test the agent's ability to search for current information
    """
    
    print("Testing Basic Search Functionality...")
    print("="*50)
    
    # Simple search query
    search_result = await agent.ainvoke({
        "messages": [("human", "Give me the latest AI news from this week, Include full URLs to source.")],
    })
    
    print("\n🔍 Search Results:")
    print(search_result["messages"][-1].content)
    
    return search_result

async def test_ecommerce_scraping(agent):
    """
    Test structured data extraction from e-commerce platforms
    """
    
    print("Testing E-commerce Data Extraction...")
    print("="*50)
    
    # Ask the agent to find and analyze a product with explicit summarization request
    ecommerce_result = await agent.ainvoke({
        "messages": [("human", "Find the top 3 wireless headphones on Amazon. Summarize ONLY: product name, price, rating, and 1-2 key features for each. Keep response under 500 words.")]
    })
    
    print("\n🛒 E-commerce Analysis:")
    print(ecommerce_result["messages"][-1].content)
    
    return ecommerce_result
async def test_social_media_simple(agent):
    """
    Test Reddit extraction with a specific approach
    """
    print("Testing Reddit Extraction...")
    print("="*50)
    
    result = await agent.ainvoke({
        "messages": [("human", "Search for 'electric vehicles on X.com' and then scrape one of the Reddit discussion pages you find. Show me what people are discussing.")]
    })
    
    print("\n📱 X.com Analysis:")
    print(result["messages"][-1].content)
    return result
async def test_complex_research(agent):
    """
    Test the agent's ability to conduct multi-step research
    """
    
    print("Testing Complex Multi-Step Research...")
    print("="*50)
    
    # Complex research query
    research_result = await agent.ainvoke({
        "messages": [("human", """
        I need to research the current state of the renewable energy market. Please:
        1. Find recent news about renewable energy developments
        2. Look up major renewable energy companies and their stock performance
        3. Analyze social media sentiment about renewable energy
        4. Provide a comprehensive market overview with key insights
        """)]
    })
    
    print("\n🔬 Complex Research Results:")
    print(research_result["messages"][-1].content)
    
    return research_result


# Execute the test
async def main():
    """Main async function to run everything"""
    # Get tools
    tools = await setup_bright_data_tools()
    print(f"✅ Tools loaded: {len(tools)}")
    
    # Create agent
    agent = await create_web_scraper_agent(tools)
    print("✅ Agent created")
    
    # Test basic search
   # search_result = await test_basic_search(agent)
   # Test ecommerce scraping
    #ecommerce_result = await test_ecommerce_scraping(agent)

# Test this version
    #social_simple = await test_social_media_simple(agent)
    # Execute the test
    research_result = await test_complex_research(agent)
    
    return tools, agent, research_result

if __name__ == "__main__":
    asyncio.run(main())

## 
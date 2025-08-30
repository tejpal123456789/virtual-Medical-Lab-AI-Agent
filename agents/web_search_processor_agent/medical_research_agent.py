# Import all necessary libraries
import asyncio
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from mcp_use.client import MCPClient
from mcp_use.adapters.langchain_adapter import LangChainAdapter
from dotenv import load_dotenv
import os
import datetime

# Load environment variables from .env
load_dotenv()

# Verify environment setup
print("✅ Medical Research Environment setup complete!")
print(f"OpenAI API Key loaded: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
print(f"Bright Data API Token loaded: {'Yes' if os.getenv('BRIGHT_DATA_API_TOKEN') else 'No'}")

async def setup_bright_data_tools():
    """
    Configure Bright Data MCP client and create LangChain-compatible tools for medical research
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
    
    print(f"✅ Connected to Bright Data MCP server for medical research")
    print(f"🏥 Available medical research tools: {len(tools)}")
    
    return tools

async def create_medical_research_agent(tools):
    """
    Create a ReAct agent configured specifically for medical research and data extraction
    """
    
    # Get current date for medical research context
    current_date = datetime.datetime.now().strftime("%B %d, %Y")
    
    # Initialize the language model optimized for medical research
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-4o",  # Use more capable model for medical accuracy
        temperature=0.1,  # Low temperature for consistent, accurate medical information
        max_tokens=4000,
        model_kwargs={
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
    )
    
    # Define comprehensive system prompt for medical research agent
    system_prompt = f"""You are an advanced Medical Research Assistant powered by web scraping capabilities. Today's date is {current_date}.

CORE MISSION: Extract, analyze, and synthesize current medical information from authoritative sources to assist healthcare professionals and researchers.

PRIORITY MEDICAL SOURCES TO TARGET:
- PubMed (pubmed.ncbi.nlm.nih.gov) - Medical literature and research papers
- ClinicalTrials.gov - Clinical trial information
- WHO (who.int) - Global health guidelines and data
- CDC (cdc.gov) - Disease surveillance and prevention guidelines  
- NIH (nih.gov) - National health institute research
- Mayo Clinic (mayoclinic.org) - Clinical expertise and patient information
- MedlinePlus (medlineplus.gov) - Reliable medical information
- New England Journal of Medicine (nejm.org) - Peer-reviewed medical research
- UpToDate (uptodate.com) - Evidence-based clinical information
- Medical journals and professional medical societies

MEDICAL RESEARCH CAPABILITIES:
✅ Literature Reviews - Search and analyze recent medical publications
✅ Clinical Trial Data - Extract trial results, protocols, and outcomes
✅ Drug Information - Dosages, interactions, side effects, contraindications
✅ Disease Information - Symptoms, diagnosis, treatment protocols
✅ Treatment Guidelines - Evidence-based treatment recommendations
✅ Epidemiological Data - Disease prevalence, statistics, trends
✅ Medical Device Information - FDA approvals, safety data, efficacy
✅ Professional Guidelines - Medical society recommendations

RESEARCH PROTOCOL:
1. IDENTIFY medical research need and classify type (literature review, clinical data, guidelines, etc.)
2. SELECT appropriate authoritative medical sources
3. EXTRACT relevant medical data using web scraping tools
4. SYNTHESIZE information with proper medical context
5. CITE sources with proper attribution
6. INCLUDE confidence levels and limitations
7. ADD medical disclaimers when appropriate

MEDICAL SAFETY GUIDELINES:
⚠️ Always include medical disclaimers for clinical information
⚠️ Emphasize that information is for educational/research purposes
⚠️ Recommend consulting healthcare professionals for medical decisions
⚠️ Flag when expert medical review is recommended
⚠️ Note evidence quality levels (systematic reviews > RCTs > observational studies)

OUTPUT FORMAT:
- Start with executive summary
- Present findings with source attribution  
- Include confidence levels and evidence quality
- Provide actionable insights for healthcare professionals
- End with appropriate medical disclaimers
- Limit responses to 2000 characters when summarizing to avoid context issues

EXAMPLE MEDICAL QUERIES YOU EXCEL AT:
• "Latest treatment protocols for Type 2 diabetes from major medical organizations"
• "Recent clinical trials on Alzheimer's disease interventions" 
• "Current WHO guidelines for infectious disease management"
• "Side effects and contraindications for [specific medication]"
• "Evidence-based diagnostic criteria for [medical condition]"
• "Recent epidemiological data on cardiovascular disease trends"

Remember: You have {len(tools)} specialized tools. Always use these tools for current medical information - never rely on potentially outdated training data for medical facts."""
    
    # Create the ReAct agent for medical research
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt
    )
    
    print("🏥 Medical Research ReAct Agent created successfully!")
    return agent

async def test_medical_literature_search(agent):
    """
    Test the agent's ability to search medical literature
    """
    
    print("Testing Medical Literature Search...")
    print("="*60)
    
    # Medical literature search query
    literature_result = await agent.ainvoke({
        "messages": [("human", "Search PubMed and other medical sources for the latest research on 'long COVID cardiovascular effects'. Provide a summary of recent findings with proper citations.")],
    })
    
    print("\n📚 Medical Literature Results:")
    print(literature_result["messages"][-1].content)
    
    return literature_result

async def test_clinical_guidelines_research(agent):
    """
    Test extraction of clinical guidelines from medical organizations
    """
    
    print("Testing Clinical Guidelines Research...")
    print("="*60)
    
    # Clinical guidelines query
    guidelines_result = await agent.ainvoke({
        "messages": [("human", "Find the latest CDC and WHO guidelines for diabetes management. Compare key recommendations and highlight any recent updates.")],
    })
    
    print("\n📋 Clinical Guidelines Results:")
    print(guidelines_result["messages"][-1].content)
    
    return guidelines_result

async def test_drug_information_research(agent):
    """
    Test drug information and pharmaceutical research
    """
    
    print("Testing Drug Information Research...")
    print("="*60)
    
    # Drug information query
    drug_result = await agent.ainvoke({
        "messages": [("human", "Research the latest information on GLP-1 receptor agonists for diabetes treatment. Include efficacy data, side effects, and any recent safety warnings from FDA or medical organizations.")],
    })
    
    print("\n💊 Drug Information Results:")
    print(drug_result["messages"][-1].content)
    
    return drug_result

async def test_clinical_trials_research(agent):
    """
    Test clinical trials data extraction
    """
    
    print("Testing Clinical Trials Research...")
    print("="*60)
    
    # Clinical trials query
    trials_result = await agent.ainvoke({
        "messages": [("human", "Search ClinicalTrials.gov and other sources for ongoing clinical trials related to 'CAR-T cell therapy for solid tumors'. Summarize trial phases, primary endpoints, and enrollment status.")],
    })
    
    print("\n🧪 Clinical Trials Results:")
    print(trials_result["messages"][-1].content)
    
    return trials_result

async def test_epidemiological_research(agent):
    """
    Test epidemiological data and public health information extraction
    """
    
    print("Testing Epidemiological Research...")
    print("="*60)
    
    # Epidemiological data query
    epi_result = await agent.ainvoke({
        "messages": [("human", "Find current epidemiological data on mental health trends post-COVID from CDC, WHO, and major health organizations. Include prevalence data, demographic breakdowns, and public health recommendations.")],
    })
    
    print("\n📊 Epidemiological Results:")
    print(epi_result["messages"][-1].content)
    
    return epi_result

async def test_comprehensive_medical_research(agent):
    """
    Test comprehensive multi-source medical research capability
    """
    
    print("Testing Comprehensive Medical Research...")
    print("="*60)
    
    # Comprehensive medical research query
    comprehensive_result = await agent.ainvoke({
        "messages": [("human", """
        Conduct a comprehensive research analysis on 'immunotherapy for melanoma treatment'. Please:
        1. Search recent medical literature and clinical studies
        2. Find current treatment guidelines from major cancer organizations
        3. Identify ongoing clinical trials and emerging therapies
        4. Analyze safety profiles and efficacy data
        5. Provide evidence-based summary with confidence levels
        
        Focus on information from the last 2 years and include proper medical disclaimers.
        """)],
    })
    
    print("\n🔬 Comprehensive Medical Research Results:")
    print(comprehensive_result["messages"][-1].content)
    
    return comprehensive_result

# Main execution function
async def main():
    """Main async function to run medical research assistant"""
    try:
        # Get tools
        tools = await setup_bright_data_tools()
        print(f"✅ Medical research tools loaded: {len(tools)}")
        
        # Create medical research agent
        agent = await create_medical_research_agent(tools)
        print("✅ Medical Research Agent created")
        
        # Run medical research tests
        print("\n🏥 Starting Medical Research Tests...")
        print("="*80)
        
        # Test 1: Medical Literature Search
        #literature_result = await test_medical_literature_search(agent)
        
        # Test 2: Clinical Guidelines Research  
        # guidelines_result = await test_clinical_guidelines_research(agent)
        
        # Test 3: Drug Information Research
        # drug_result = await test_drug_information_research(agent)
        
        # Test 4: Clinical Trials Research
        # trials_result = await test_clinical_trials_research(agent)
        
        # Test 5: Epidemiological Research
        # epi_result = await test_epidemiological_research(agent)
        
        # Test 6: Comprehensive Medical Research
        comprehensive_result = await test_comprehensive_medical_research(agent)
        
        print("\n✅ Medical Research Assistant testing completed!")
        
        return tools, agent,comprehensive_result
        
    except Exception as e:
        print(f"❌ Error in medical research assistant: {str(e)}")
        raise

if __name__ == "__main__":
    print("🏥 Starting Medical Research Assistant with Bright Data MCP...")
    asyncio.run(main())
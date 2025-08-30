"""
Agent Decision System for Multi-Agent Medical Chatbot

This module handles the orchestration of different agents using LangGraph.
It dynamically routes user queries to the appropriate agent based on content and context.
"""

import json
from typing import Dict, List, Optional, Any, Literal, TypedDict, Union, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import MessagesState, StateGraph, END
import os, getpass
from dotenv import load_dotenv
from agents.rag_agent import MedicalRAG
from agents.web_search_processor_agent import WebSearchProcessorAgent
from agents.image_analysis_agent import ImageAnalysisAgent
from agents.guardrails.local_guardrails import LocalGuardrails

# Import long-term memory components
from memory import LongTermMemoryManager, MemoryEnhancedPromptBuilder
from memory.memory_utils import extract_user_preferences, should_store_as_medical_history
from memory.fallback_memory import FallbackMemoryManager

from langgraph.checkpoint.memory import MemorySaver

import cv2
import numpy as np

# Add imports for medical research functionality
import asyncio
import datetime
from langchain_openai import ChatOpenAI
from mcp_use.client import MCPClient
from mcp_use.adapters.langchain_adapter import LangChainAdapter
from langgraph.prebuilt import create_react_agent

from config import Config
import qualifire


load_dotenv()

# Load configuration
config = Config()

# Initialize short-term memory (LangGraph checkpointer)
memory = MemorySaver()

# Specify a thread
thread_config = {"configurable": {"thread_id": "1"}}

# Initialize long-term memory manager (global instance)
long_term_memory = None
memory_prompt_builder = None
qualifire.init(
        api_key=os.getenv("QUALIFIRE_API_KEY"),
    )
# Medical Research Tools Setup
async def setup_bright_data_tools():
    """
    Configure Bright Data MCP client for medical research with specialized capabilities
    """
    
    # Configure MCP server for medical research
    bright_data_config = {
        "mcpServers": {
            "Bright Data Medical": {
                "command": "npx",
                "args": ["@brightdata/mcp"],
                "env": {
                    "API_TOKEN": os.getenv("BRIGHT_DATA_API_TOKEN"),
                }
            }
        }
    }
    
    # Create client and convert to LangChain tools
    client = MCPClient.from_dict(bright_data_config)
    adapter = LangChainAdapter()
    tools = await adapter.create_tools(client)
    
    print(f"✅ Medical Research Tools Configured: {len(tools)} available")
    
    return tools

async def create_medical_research_agent(tools):
    """
    Create a sophisticated ReAct agent optimized for medical research
    """
    
    current_date = datetime.datetime.now().strftime("%B %d, %Y")
    
    # Initialize GPT-4o for medical accuracy
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.1,  # Low temperature for consistent medical information
        max_tokens=4000,
    )
    
    # The comprehensive medical research system prompt
    system_prompt = f"""You are an Advanced Medical Research Assistant with specialized web scraping capabilities. Today's date is {current_date}.

🎯 CORE MISSION: Extract, analyze, and synthesize current medical information from authoritative sources to assist healthcare professionals and researchers.

🏥 PRIORITY MEDICAL SOURCES TO TARGET:
• PubMed (pubmed.ncbi.nlm.nih.gov) - Medical literature and peer-reviewed research
• ClinicalTrials.gov - Clinical trial protocols, results, and recruitment status
• WHO (who.int) - Global health guidelines and epidemiological data
• CDC (cdc.gov) - Disease surveillance, prevention guidelines, and public health data
• NIH (nih.gov) - National health institute research and funding information
• Mayo Clinic (mayoclinic.org) - Clinical expertise and evidence-based patient information
• MedlinePlus (medlineplus.gov) - Consumer health information and medical encyclopedia
• New England Journal of Medicine (nejm.org) - High-impact medical research
• UpToDate (uptodate.com) - Evidence-based clinical decision support
• FDA (fda.gov) - Drug approvals, safety alerts, and medical device information

⚡ ADVANCED RESEARCH CAPABILITIES:
✅ **Systematic Literature Reviews** - Multi-database search and meta-analysis synthesis
✅ **Clinical Trial Intelligence** - Protocol analysis, enrollment tracking, outcome reporting
✅ **Pharmacological Research** - Drug mechanisms, interactions, contraindications, dosing
✅ **Disease Profiling** - Pathophysiology, diagnostic criteria, differential diagnosis
✅ **Treatment Protocol Analysis** - Evidence-based guidelines, comparative effectiveness
✅ **Epidemiological Intelligence** - Population health trends, risk factors, prevention strategies

🔬 INTELLIGENT RESEARCH PROTOCOL:
1. **QUERY ANALYSIS** - Classify medical research type and determine optimal search strategy
2. **SOURCE PRIORITIZATION** - Target highest-quality, most authoritative medical sources
3. **MULTI-SOURCE EXTRACTION** - Systematically gather data from complementary sources  
4. **EVIDENCE SYNTHESIS** - Integrate findings with proper medical context and hierarchy
5. **CLINICAL CORRELATION** - Connect research findings to practical clinical applications

⚠️ MEDICAL SAFETY & COMPLIANCE FRAMEWORK:
🛡️ Always include appropriate medical disclaimers for clinical information
🛡️ Emphasize educational/research purpose limitations
🛡️ Recommend healthcare professional consultation for medical decisions
🛡️ Flag when specialized medical expertise is required

You have {len(tools)} specialized research tools at your disposal. Always utilize these tools for current medical information."""
    
    # Create the specialized medical research agent
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt
    )
    
    print("🏥 Advanced Medical Research Agent Successfully Initialized!")
    return agent

# Agent that takes the decision of routing the request further to correct task specific agent
class AgentConfig:
    """Configuration settings for the agent decision system."""
    
    # Decision model
    DECISION_MODEL = "gpt-4o"  # or whichever model you prefer
    
    # Vision model for image analysis
    VISION_MODEL = "gpt-4o"
    
    # Confidence threshold for responses
    CONFIDENCE_THRESHOLD = 0.85
    
    # System instructions for the decision agent
    DECISION_SYSTEM_PROMPT = """You are an intelligent medical triage system that routes user queries to 
    the appropriate specialized agent. Your job is to analyze the user's request and determine which agent 
    is best suited to handle it based on the query content, presence of images, and conversation context.

    Available agents:
    1. CONVERSATION_AGENT - For general chat, greetings, and non-medical questions.
    2. RAG_AGENT - For specific medical knowledge questions that can be answered from established medical literature. Currently ingested medical knowledge involves 'introduction to brain tumor', 'deep learning techniques to diagnose and detect brain tumors', 'deep learning techniques to diagnose and detect covid / covid-19 from chest x-ray'.
    3. WEB_SEARCH_PROCESSOR_AGENT - For questions about recent medical developments, current outbreaks, or time-sensitive medical information.
    4. BRAIN_TUMOR_AGENT - For analysis of brain MRI images to detect and segment tumors.
    5. CHEST_XRAY_AGENT - For analysis of chest X-ray images to detect abnormalities.
    6. SKIN_LESION_AGENT - For analysis of skin lesion images to classify them as benign or malignant.
    7. MEDICAL_RESEARCH_AGENT - For conducting medical research to particular medical topic 

    Make your decision based on these guidelines:
    - If the user has not uploaded any image, always route to the conversation agent.
    - If the user uploads a medical image, decide which medical vision agent is appropriate based on the image type and the user's query. If the image is uploaded without a query, always route to the correct medical vision agent based on the image type.
    - If the user asks about recent medical developments or current health situations, use the web search processor agent.
    - If the user asks specific medical knowledge questions that can be answered from existing literature, use the RAG agent.
    - If the user asks for comprehensive medical research, literature review, clinical trial information, or in-depth analysis of a medical topic that requires web scraping from medical sources (PubMed, clinical trials, medical journals), use the MEDICAL_RESEARCH_AGENT.
    - For general conversation, greetings, or non-medical questions, use the conversation agent. But if image is uploaded, always go to the medical vision agents first.

    Key distinction between WEB_SEARCH_PROCESSOR_AGENT and MEDICAL_RESEARCH_AGENT:
    - WEB_SEARCH_PROCESSOR_AGENT: For current news, outbreak information, and general medical updates
    - MEDICAL_RESEARCH_AGENT: For comprehensive research requiring deep analysis of medical literature, clinical trials, and authoritative medical sources

    You must provide your answer in JSON format with the following structure:
    {{
    "agent": "AGENT_NAME",
    "reasoning": "Your step-by-step reasoning for selecting this agent",
    "confidence": 0.95  // Value between 0.0 and 1.0 indicating your confidence in this decision
    }}
    """

    image_analyzer = ImageAnalysisAgent(config=config)


class AgentState(MessagesState):
    """State maintained across the workflow."""
    # messages: List[BaseMessage]  # Conversation history
    agent_name: Optional[str]  # Current active agent
    current_input: Optional[Union[str, Dict]]  # Input to be processed
    has_image: bool  # Whether the current input contains an image
    image_type: Optional[str]  # Type of medical image if present
    output: Optional[str]  # Final output to user
    needs_human_validation: bool  # Whether human validation is required
    retrieval_confidence: float  # Confidence in retrieval (for RAG agent)
    bypass_routing: bool  # Flag to bypass agent routing for guardrails
    insufficient_info: bool  # Flag indicating RAG response has insufficient information
    user_id: Optional[str]  # User ID for long-term memory tracking
    memory_enhanced: bool  # Flag indicating if memory enhancement was applied


class AgentDecision(TypedDict):
    """Output structure for the decision agent."""
    agent: str
    reasoning: str
    confidence: float


def create_agent_graph():
    """Create and configure the LangGraph for agent orchestration."""

    # Initialize guardrails with the same LLM used elsewhere
    guardrails = LocalGuardrails(config.rag.llm)
    
    # Initialize long-term memory system
    global long_term_memory, memory_prompt_builder
    if long_term_memory is None:
        try:
            long_term_memory = LongTermMemoryManager(config)
            memory_prompt_builder = MemoryEnhancedPromptBuilder(long_term_memory)
            print("✅ Long-term memory system initialized")
        except Exception as e:
            print(f"⚠️ Long-term memory initialization failed: {e}")
            print("🔄 Attempting fallback memory system...")
            try:
                long_term_memory = FallbackMemoryManager(config)
                memory_prompt_builder = MemoryEnhancedPromptBuilder(long_term_memory)
                print("✅ Fallback memory system initialized")
            except Exception as e2:
                print(f"❌ Fallback memory also failed: {e2}")
                long_term_memory = None
                memory_prompt_builder = None

    # LLM
    decision_model = config.agent_decision.llm
    
    # Initialize the output parser
    json_parser = JsonOutputParser(pydantic_object=AgentDecision)
    
    # Create the decision prompt
    decision_prompt = ChatPromptTemplate.from_messages([
        ("system", AgentConfig.DECISION_SYSTEM_PROMPT),
        ("human", "{input}")
    ])
    
    # Create the decision chain
    decision_chain = decision_prompt | decision_model | json_parser
    
    # Define graph state transformations
    def analyze_input(state: AgentState) -> AgentState:
        """Analyze the input to detect images and determine input type."""
        print("state", state)
        current_input = state["current_input"]
        print("current_input", current_input)
        has_image = False
        image_type = None
        
        # Extract or generate user ID for long-term memory tracking
        user_id = state.get("user_id") or "default_user"
        
        # Get the text from the input
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Check input through guardrails if text is present
        if input_text:
            is_allowed, message = guardrails.check_input(input_text)
            if not is_allowed:
                # If input is blocked, return early with guardrail message
                print(f"Selected agent: INPUT GUARDRAILS, Message: ", message)
                return {
                    **state,
                    "messages": message,
                    "agent_name": "INPUT_GUARDRAILS",
                    "has_image": False,
                    "image_type": None,
                    "user_id": user_id,
                    "memory_enhanced": False,
                    "bypass_routing": True  # flag to end flow
                }
        
        # Original image processing code
        if isinstance(current_input, dict) and "image" in current_input:
            has_image = True
            image_path = current_input.get("image", None)
            image_type_response = AgentConfig.image_analyzer.analyze_image(image_path)
            image_type = image_type_response['image_type']
            print("ANALYZED IMAGE TYPE: ", image_type)
        
        return {
            **state,
            "has_image": has_image,
            "image_type": image_type,
            "user_id": user_id,
            "memory_enhanced": False,
            "bypass_routing": False  # Explicitly set to False for normal flow
        }
    
    def check_if_bypassing(state: AgentState) -> str:
        """Check if we should bypass normal routing due to guardrails."""
        if state.get("bypass_routing", False):
            return "apply_guardrails"
        return "route_to_agent"
    
    def route_to_agent(state: AgentState) -> Dict:
        """Make decision about which agent should handle the query."""
        messages = state["messages"]
        current_input = state["current_input"]
        has_image = state["has_image"]
        image_type = state["image_type"]
        
        # Prepare input for decision model
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Create context from recent conversation history (last 3 messages)
        recent_context = ""
        for msg in messages[-6:]:  # Get last 3 exchanges (6 messages)  # Not provided control from config
            if isinstance(msg, HumanMessage):
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                recent_context += f"Assistant: {msg.content}\n"
        
        # Combine everything for the decision input
        decision_input = f"""
        User query: {input_text}

        Recent conversation context:
        {recent_context}

        Has image: {has_image}
        Image type: {image_type if has_image else 'None'}

        Based on this information, which agent should handle this query?
        """
        
        # Make the decision
        decision = decision_chain.invoke({"input": decision_input})

        # Decided agent
        print(f"Decision: {decision['agent']}")
        
        # Update state with decision
        updated_state = {
            **state,
            "agent_name": decision["agent"],
        }
        
        # Route based on agent name and confidence
        if decision["confidence"] < AgentConfig.CONFIDENCE_THRESHOLD:
            return {"agent_state": updated_state, "next": "needs_validation"}
        
        return {"agent_state": updated_state, "next": decision["agent"]}

    # Define agent execution functions (these will be implemented in their respective modules)
    def run_conversation_agent(state: AgentState) -> AgentState:
        """Handle general conversation with long-term memory enhancement."""

        print(f"Selected agent: CONVERSATION_AGENT")

        messages = state["messages"]
        current_input = state["current_input"]
        user_id = state.get("user_id", "default_user")
        
        # Prepare input for decision model
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Create context from recent conversation history
        recent_context = ""
        for msg in messages:#[-20:]:  # Get last 10 exchanges (20 messages)  # currently considering complete history - limit control from config
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"
        
        # Base conversation prompt
        base_conversation_prompt = f"""User query: {input_text}

        Recent conversation context: {recent_context}

        You are an AI-powered Medical Conversation Assistant. Your goal is to facilitate smooth and informative conversations with users, handling both casual and medical-related queries. You must respond naturally while ensuring medical accuracy and clarity.

        ### Role & Capabilities
        - Engage in **general conversation** while maintaining professionalism.
        - Answer **medical questions** using verified knowledge.
        - Route **complex queries** to RAG (retrieval-augmented generation) or web search if needed.
        - Handle **follow-up questions** while keeping track of conversation context.
        - Redirect **medical images** to the appropriate AI analysis agent.

        ### Guidelines for Responding:
        1. **General Conversations:**
        - If the user engages in casual talk (e.g., greetings, small talk), respond in a friendly, engaging manner.
        - Keep responses **concise and engaging**, unless a detailed answer is needed.

        2. **Medical Questions:**
        - If you have **high confidence** in answering, provide a medically accurate response.
        - Ensure responses are **clear, concise, and factual**.

        3. **Follow-Up & Clarifications:**
        - Maintain conversation history for better responses.
        - If a query is unclear, ask **follow-up questions** before answering.

        4. **Handling Medical Image Analysis:**
        - Do **not** attempt to analyze images yourself.
        - If user speaks about analyzing or processing or detecting or segmenting or classifying any disease from any image, ask the user to upload the image so that in the next turn it is routed to the appropriate medical vision agents.
        - If an image was uploaded, it would have been routed to the medical computer vision agents. Read the history to know about the diagnosis results and continue conversation if user asks anything regarding the diagnosis.
        - After processing, **help the user interpret the results**.

        5. **Uncertainty & Ethical Considerations:**
        - If unsure, **never assume** medical facts.
        - Recommend consulting a **licensed healthcare professional** for serious medical concerns.
        - Avoid providing **medical diagnoses** or **prescriptions**—stick to general knowledge.

        ### Response Format:
        - Maintain a **conversational yet professional tone**.
        - Use **bullet points or numbered lists** for clarity when needed.
        - If pulling from external sources (RAG/Web Search), mention **where the information is from** (e.g., "According to Mayo Clinic...").
        - If a user asks for a diagnosis, remind them to **seek medical consultation**.

        ### Example User Queries & Responses:

        **User:** "Hey, how's your day going?"
        **You:** "I'm here and ready to help! How can I assist you today?"

        **User:** "I have a headache and fever. What should I do?"
        **You:** "I'm not a doctor, but headaches and fever can have various causes, from infections to dehydration. If your symptoms persist, you should see a medical professional."

        Conversational LLM Response:"""

        # Store important personal info BEFORE processing to make it available immediately
        if long_term_memory:
            try:
                # More precise keyword matching to avoid false positives
                input_lower = input_text.lower()
                is_identity_statement = (
                    input_lower.startswith("my name is") or
                    input_lower.startswith("i am") or 
                    input_lower.startswith("i'm") or
                    input_lower.startswith("call me") or
                    " my name is " in input_lower
                )
                
                # Make sure it's not a question about their name
                is_question = any(q in input_lower for q in [
                    "what is my name", "what's my name", "do you know my name",
                    "can you tell me my name", "remember my name"
                ])
                
                if is_identity_statement and not is_question:
                    # Store name information immediately for next query
                    name_info = f"User introduced themselves: {input_text}"
                    long_term_memory.store_medical_insight(
                        user_id=user_id,
                        insight=name_info,
                        source_agent="CONVERSATION_AGENT",
                        confidence=1.0,
                        metadata={
                            "info_type": "user_identity", 
                            "immediate_storage": True,
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                    )
                    print(f"✅ Pre-stored user identity: {input_text[:30]}...")
            except Exception as e:
                print(f"⚠️ Failed to pre-store identity info: {e}")

        # Enhance prompt with long-term memory if available
        if long_term_memory and memory_prompt_builder and config.memory.enhance_conversation_prompts:
            try:
                conversation_prompt = memory_prompt_builder.enhance_conversation_prompt(
                    user_id=user_id,
                    query=input_text,
                    base_prompt=base_conversation_prompt
                )
                memory_enhanced = True
                print("✅ Enhanced conversation prompt with long-term memory")
            except Exception as e:
                print(f"⚠️ Memory enhancement failed, using base prompt: {e}")
                conversation_prompt = base_conversation_prompt
                memory_enhanced = False
        else:
            conversation_prompt = base_conversation_prompt
            memory_enhanced = False

        # print("Conversation Prompt:", conversation_prompt)

        response = config.conversation.llm.invoke(conversation_prompt)

        # Store interaction in long-term memory BEFORE responding if it contains important info
        if long_term_memory:
            try:
                # Check if this message contains user identity or important personal info
                input_lower = input_text.lower()
                is_personal_info = (
                    input_lower.startswith("my name is") or
                    input_lower.startswith("i am") or 
                    input_lower.startswith("i have") or
                    input_lower.startswith("i suffer from") or
                    input_lower.startswith("i was diagnosed") or
                    " my name is " in input_lower
                )
                
                # Avoid storing questions as personal info
                is_question = any(q in input_lower for q in [
                    "what is", "what's", "do you know", "can you tell me", "remember"
                ])
                
                if is_personal_info and not is_question:
                    # Store important personal information immediately
                    personal_info = f"User identity/personal info: {input_text}"
                    long_term_memory.store_medical_insight(
                        user_id=user_id,
                        insight=personal_info,
                        source_agent="CONVERSATION_AGENT",
                        confidence=0.9,
                        metadata={"info_type": "personal_identity", "immediate_storage": True}
                    )
                    print(f"✅ Stored personal information immediately: {input_text[:50]}...")
                
                # Also store regular interaction
                long_term_memory.update_memory_from_interaction(
                    user_id=user_id,
                    user_message=input_text,
                    agent_response=response.content,
                    agent_name="CONVERSATION_AGENT"
                )
            except Exception as e:
                print(f"⚠️ Failed to store conversation in long-term memory: {e}")

        # print("Conversation respone:", response)

        # response = AIMessage(content="This would be handled by the conversation agent.")

        return {
            **state,
            "output": response,
            "agent_name": "CONVERSATION_AGENT",
            "memory_enhanced": memory_enhanced
        }
    def run_medical_research_agent(state: AgentState) -> AgentState:
        """Handle medical research queries using Bright Data tools with long-term memory enhancement."""

        print(f"Selected agent: MEDICAL_RESEARCH_AGENT")

        messages = state["messages"]
        current_input = state["current_input"]
        user_id = state.get("user_id", "default_user")
        
        # Prepare input text
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")

        # Enhance research query with long-term memory if available
        enhanced_research_query = input_text
        memory_enhanced = False
        if long_term_memory and memory_prompt_builder and config.memory.enhance_research_prompts:
            try:
                # Get relevant research history
                research_memories = long_term_memory.retrieve_relevant_memories(
                    user_id=user_id,
                    query=input_text,
                    limit=3
                )
                
                if research_memories:
                    memory_context = "Previous research context: "
                    for memory in research_memories:
                        # Handle both dict and string memory objects
                        if isinstance(memory, dict):
                            memory_content = memory.get("memory", str(memory))
                        else:
                            memory_content = str(memory)
                        memory_context += f"{memory_content}; "
                    
                    enhanced_research_query = f"{input_text}. {memory_context}"
                    memory_enhanced = True
                    print("✅ Enhanced research query with long-term memory context")
            except Exception as e:
                print(f"⚠️ Research memory enhancement failed: {e}")

        async def conduct_research():
            """Async function to conduct medical research"""
            try:
                # Initialize medical research tools
                tools = await setup_bright_data_tools()
                print(f"✅ Medical Research Tools Loaded: {len(tools)}")
                
                # Create specialized medical agent
                agent = await create_medical_research_agent(tools)
                print("✅ Medical Research Agent Ready")
                
                # Execute comprehensive research
                research_result = await agent.ainvoke({
                    "messages": [("human", f"""
                    Conduct comprehensive medical research on: {enhanced_research_query}
                    
                    Please:
                    1. Search recent medical literature and clinical studies
                    2. Find current treatment guidelines from major medical organizations  
                    3. Identify relevant clinical trials and emerging therapies
                    4. Analyze safety profiles and efficacy data
                    5. Provide evidence-based summary with proper citations
                    
                    Focus on authoritative sources and include appropriate medical disclaimers.
                    """)]
                })
                
                research_content = research_result["messages"][-1].content
                print(f"✅ Medical research completed ({len(research_content)} characters)")
                
                return research_content
                
            except Exception as e:
                print(f"❌ Medical research failed: {e}")
                return f"I apologize, but I encountered an error while conducting medical research: {str(e)}. Please try again or rephrase your research query."

        # Run the async research function following bright_data.py pattern
        try:
            research_content = asyncio.run(conduct_research())
            response = AIMessage(content=research_content)
            
            # Store research results in long-term memory
            if long_term_memory:
                try:
                    long_term_memory.store_medical_insight(
                        user_id=user_id,
                        insight=f"Medical research on: {input_text} | Key findings: {research_content[:300]}...",
                        source_agent="MEDICAL_RESEARCH_AGENT",
                        confidence=0.9,  # High confidence for research results
                        metadata={"research_type": "comprehensive", "query_length": len(input_text)}
                    )
                    
                    long_term_memory.update_memory_from_interaction(
                        user_id=user_id,
                        user_message=input_text,
                        agent_response=research_content,
                        agent_name="MEDICAL_RESEARCH_AGENT"
                    )
                    print("✅ Stored research results in long-term memory")
                except Exception as e:
                    print(f"⚠️ Failed to store research in long-term memory: {e}")
            
        except Exception as e:
            print(f"❌ Research execution failed: {e}")
            response = AIMessage(content=f"Medical research system encountered an error: {str(e)}")
            memory_enhanced = False

        return {
            **state,
            "output": response,
            "agent_name": "MEDICAL_RESEARCH_AGENT",
            "memory_enhanced": memory_enhanced
        }

    
    def run_rag_agent(state: AgentState) -> AgentState:
        """Handle medical knowledge queries using RAG with long-term memory enhancement."""
        # Initialize the RAG agent

        print(f"Selected agent: RAG_AGENT")

        rag_agent = MedicalRAG(config)
        
        messages = state["messages"]
        query = state["current_input"]
        user_id = state.get("user_id", "default_user")
        rag_context_limit = config.rag.context_limit

        recent_context = ""
        for msg in messages[-rag_context_limit:]:# limit controlled from config
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"

        # Enhance query with long-term memory context if available
        enhanced_query = query
        memory_enhanced = False
        if long_term_memory and config.memory.enhance_rag_prompts:
            try:
                # Get relevant medical memories for context
                relevant_memories = long_term_memory.get_contextual_memories_for_agent(
                    user_id=user_id,
                    agent_name="RAG_AGENT",
                    current_query=str(query),
                    limit=3
                )
                
                if relevant_memories:
                    memory_context = "Previous medical context: "
                    for memory in relevant_memories:
                        # Handle both dict and string memory objects
                        if isinstance(memory, dict):
                            memory_content = memory.get("memory", str(memory))
                        else:
                            memory_content = str(memory)
                        memory_context += f"{memory_content}; "
                    
                    enhanced_query = f"{query}. {memory_context}"
                    memory_enhanced = True
                    print("✅ Enhanced RAG query with long-term memory context")
            except Exception as e:
                print(f"⚠️ RAG memory enhancement failed: {e}")

        response = rag_agent.process_query(enhanced_query, chat_history=recent_context)
        retrieval_confidence = response.get("confidence", 0.0)  # Default to 0.0 if not provided

        print(f"Retrieval Confidence: {retrieval_confidence}")
        print(f"Sources: {len(response['sources'])}")

        # Check if response indicates insufficient information
        insufficient_info = False
        response_content = response["response"]
        
        # Extract the content properly based on type
        if isinstance(response_content, dict) and hasattr(response_content, 'content'):
            # If it's an AIMessage or similar object with a content attribute
            response_text = response_content.content
        else:
            # If it's already a string
            response_text = response_content
            
        print(f"Response text type: {type(response_text)}")
        print(f"Response text preview: {response_text[:100]}...")
        
        if isinstance(response_text, str) and (
            "I don't have enough information to answer this question based on the provided context" in response_text or 
            "I don't have enough information" in response_text or 
            "don't have enough information" in response_text.lower() or
            "not enough information" in response_text.lower() or
            "insufficient information" in response_text.lower() or
            "cannot answer" in response_text.lower() or
            "unable to answer" in response_text.lower()
            ):
            
            print("RAG response indicates insufficient information")
            print(f"Response text that triggered insufficient_info: {response_text[:100]}...")
            insufficient_info = True

        print(f"Insufficient info flag set to: {insufficient_info}")

        # Store interaction in long-term memory if medical content
        if long_term_memory and isinstance(response_text, str):
            try:
                # Store medical insights from RAG response
                if should_store_as_medical_history(str(query), response_text):
                    long_term_memory.store_medical_insight(
                        user_id=user_id,
                        insight=f"Medical query: {query} | Response: {response_text[:200]}...",
                        source_agent="RAG_AGENT",
                        confidence=retrieval_confidence,
                        metadata={"sources_count": len(response['sources'])}
                    )
                
                # Update general interaction memory
                long_term_memory.update_memory_from_interaction(
                    user_id=user_id,
                    user_message=str(query),
                    agent_response=response_text,
                    agent_name="RAG_AGENT",
                    metadata={"confidence": retrieval_confidence}
                )
            except Exception as e:
                print(f"⚠️ Failed to store RAG interaction in long-term memory: {e}")

        # Store RAG output ONLY if confidence is high
        if retrieval_confidence >= config.rag.min_retrieval_confidence:
            # response_output = response["response"]
            response_output = AIMessage(content=response_text)
        else:
            response_output = AIMessage(content="")
        
        return {
            **state,
            "output": response_output,
            "needs_human_validation": False,  # Assuming no validation needed for RAG responses
            "retrieval_confidence": retrieval_confidence,
            "agent_name": "RAG_AGENT",
            "insufficient_info": insufficient_info,
            "memory_enhanced": memory_enhanced
        }

    # Web Search Processor Node
    def run_web_search_processor_agent(state: AgentState) -> AgentState:
        """Handles web search results, processes them with LLM, and generates a refined response."""

        print(f"Selected agent: WEB_SEARCH_PROCESSOR_AGENT")
        print("[WEB_SEARCH_PROCESSOR_AGENT] Processing Web Search Results...")
        
        messages = state["messages"]
        user_id = state.get("user_id", "default_user")
        web_search_context_limit = config.web_search.context_limit

        recent_context = ""
        for msg in messages[-web_search_context_limit:]: # limit controlled from config
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"

        web_search_processor = WebSearchProcessorAgent(config)

        # Get query text for memory enhancement
        query_text = ""
        if isinstance(state["current_input"], str):
            query_text = state["current_input"]
        elif isinstance(state["current_input"], dict):
            query_text = state["current_input"].get("text", "")

        # Enhance with long-term memory context if available
        memory_enhanced = False
        if long_term_memory:
            try:
                # Get relevant memories for web search context
                relevant_memories = long_term_memory.get_contextual_memories_for_agent(
                    user_id=user_id,
                    agent_name="WEB_SEARCH_PROCESSOR_AGENT",
                    current_query=query_text,
                    limit=2
                )
                
                if relevant_memories:
                    memory_context = "Previous search context: "
                    for memory in relevant_memories:
                        # Handle both dict and string memory objects
                        if isinstance(memory, dict):
                            memory_content = memory.get("memory", str(memory))
                        else:
                            memory_content = str(memory)
                        memory_context += f"{memory_content}; "
                    
                    # Add memory context to chat history
                    recent_context += f"\nLong-term memory context: {memory_context}\n"
                    memory_enhanced = True
                    print("✅ Enhanced web search with long-term memory context")
            except Exception as e:
                print(f"⚠️ Web search memory enhancement failed: {e}")

        processed_response = web_search_processor.process_web_search_results(query=state["current_input"], chat_history=recent_context)

        # print("######### DEBUG WEB SEARCH:", processed_response)
        
        # Store web search results in long-term memory
        if long_term_memory:
            try:
                response_content = processed_response.content if hasattr(processed_response, 'content') else str(processed_response)
                
                long_term_memory.update_memory_from_interaction(
                    user_id=user_id,
                    user_message=query_text,
                    agent_response=response_content,
                    agent_name="WEB_SEARCH_PROCESSOR_AGENT",
                    metadata={"search_type": "current_information"}
                )
                print("✅ Stored web search results in long-term memory")
            except Exception as e:
                print(f"⚠️ Failed to store web search in long-term memory: {e}")
        
        if state['agent_name'] != None:
            involved_agents = f"{state['agent_name']}, WEB_SEARCH_PROCESSOR_AGENT"
        else:
            involved_agents = "WEB_SEARCH_PROCESSOR_AGENT"

        # Overwrite any previous output with the processed Web Search response
        return {
            **state,
            # "output": "This would be handled by the web search agent, finding the latest information.",
            "output": processed_response,
            "agent_name": involved_agents,
            "memory_enhanced": memory_enhanced
        }

    # Define Routing Logic
    def confidence_based_routing(state: AgentState) -> Dict[str, str]:
        """Route based on RAG confidence score and response content."""
        # Debug prints
        print(f"Routing check - Retrieval confidence: {state.get('retrieval_confidence', 0.0)}")
        print(f"Routing check - Insufficient info flag: {state.get('insufficient_info', False)}")
        
        # Redirect if confidence is low or if response indicates insufficient info
        if (state.get("retrieval_confidence", 0.0) < config.rag.min_retrieval_confidence or 
            state.get("insufficient_info", False)):
            print("Re-routed to Web Search Agent due to low confidence or insufficient information...")
            return "WEB_SEARCH_PROCESSOR_AGENT"  # Correct format
        return "check_validation"  # No transition needed if confidence is high and info is sufficient
    
    def run_brain_tumor_agent(state: AgentState) -> AgentState:
        """Handle brain MRI image analysis with long-term memory storage."""

        print(f"Selected agent: BRAIN_TUMOR_AGENT")
        
        current_input = state["current_input"]
        user_id = state.get("user_id", "default_user")

        response = AIMessage(content="This would be handled by the brain tumor agent, analyzing the MRI image.")

        # Store diagnostic result in long-term memory
        if long_term_memory:
            try:
                long_term_memory.store_medical_insight(
                    user_id=user_id,
                    insight="Brain MRI analysis for tumor detection performed",
                    source_agent="BRAIN_TUMOR_AGENT",
                    confidence=0.8,  # AI diagnostic confidence
                    metadata={
                        "image_type": "brain_mri",
                        "analysis_date": datetime.datetime.now().isoformat()
                    }
                )
                
                long_term_memory.update_memory_from_interaction(
                    user_id=user_id,
                    user_message="Uploaded brain MRI for tumor analysis",
                    agent_response=response.content,
                    agent_name="BRAIN_TUMOR_AGENT",
                    metadata={"medical_imaging": True}
                )
                print("✅ Stored brain tumor analysis in long-term memory")
            except Exception as e:
                print(f"⚠️ Failed to store brain tumor analysis in memory: {e}")

        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "BRAIN_TUMOR_AGENT",
            "memory_enhanced": True
        }
    
    def run_chest_xray_agent(state: AgentState) -> AgentState:
        """Handle chest X-ray image analysis with long-term memory storage."""

        current_input = state["current_input"]
        image_path = current_input.get("image", None)
        user_id = state.get("user_id", "default_user")

        print(f"Selected agent: CHEST_XRAY_AGENT")

        # classify chest x-ray into covid or normal
        predicted_class = AgentConfig.image_analyzer.classify_chest_xray(image_path)

        if predicted_class == "covid19":
            response = AIMessage(content="The analysis of the uploaded chest X-ray image indicates a **POSITIVE** result for **COVID-19**.")
            diagnosis_result = "COVID-19 POSITIVE"
        elif predicted_class == "normal":
            response = AIMessage(content="The analysis of the uploaded chest X-ray image indicates a **NEGATIVE** result for **COVID-19**, i.e., **NORMAL**.")
            diagnosis_result = "COVID-19 NEGATIVE (Normal)"
        else:
            response = AIMessage(content="The uploaded image is not clear enough to make a diagnosis / the image is not a medical image.")
            diagnosis_result = "Inconclusive - Image quality insufficient"

        # Store diagnostic result in long-term memory
        if long_term_memory:
            try:
                long_term_memory.store_medical_insight(
                    user_id=user_id,
                    insight=f"Chest X-ray analysis: {diagnosis_result}",
                    source_agent="CHEST_XRAY_AGENT",
                    confidence=0.8,  # AI diagnostic confidence
                    metadata={
                        "image_type": "chest_xray",
                        "diagnosis": diagnosis_result,
                        "predicted_class": predicted_class,
                        "analysis_date": datetime.datetime.now().isoformat()
                    }
                )
                
                long_term_memory.update_memory_from_interaction(
                    user_id=user_id,
                    user_message="Uploaded chest X-ray for COVID-19 analysis",
                    agent_response=response.content,
                    agent_name="CHEST_XRAY_AGENT",
                    metadata={"medical_imaging": True}
                )
                print("✅ Stored chest X-ray diagnosis in long-term memory")
            except Exception as e:
                print(f"⚠️ Failed to store chest X-ray diagnosis in memory: {e}")

        # response = AIMessage(content="This would be handled by the chest X-ray agent, analyzing the image.")

        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "CHEST_XRAY_AGENT",
            "memory_enhanced": True
        }
    
    def run_skin_lesion_agent(state: AgentState) -> AgentState:
        """Handle skin lesion image analysis with long-term memory storage."""

        current_input = state["current_input"]
        image_path = current_input.get("image", None)
        user_id = state.get("user_id", "default_user")

        print(f"Selected agent: SKIN_LESION_AGENT")

        # classify chest x-ray into covid or normal
        predicted_mask = AgentConfig.image_analyzer.segment_skin_lesion(image_path)

        if predicted_mask:
            response = AIMessage(content="Following is the analyzed **segmented** output of the uploaded skin lesion image:")
            diagnosis_result = "Skin lesion segmentation completed"
        else:
            response = AIMessage(content="The uploaded image is not clear enough to make a diagnosis / the image is not a medical image.")
            diagnosis_result = "Inconclusive - Image quality insufficient"

        # Store diagnostic result in long-term memory
        if long_term_memory:
            try:
                long_term_memory.store_medical_insight(
                    user_id=user_id,
                    insight=f"Skin lesion analysis: {diagnosis_result}",
                    source_agent="SKIN_LESION_AGENT",
                    confidence=0.8,  # AI diagnostic confidence
                    metadata={
                        "image_type": "skin_lesion",
                        "segmentation_success": bool(predicted_mask),
                        "analysis_date": datetime.datetime.now().isoformat()
                    }
                )
                
                long_term_memory.update_memory_from_interaction(
                    user_id=user_id,
                    user_message="Uploaded skin lesion image for analysis",
                    agent_response=response.content,
                    agent_name="SKIN_LESION_AGENT",
                    metadata={"medical_imaging": True}
                )
                print("✅ Stored skin lesion analysis in long-term memory")
            except Exception as e:
                print(f"⚠️ Failed to store skin lesion analysis in memory: {e}")

        # response = AIMessage(content="This would be handled by the skin lesion agent, analyzing the skin image.")

        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "SKIN_LESION_AGENT",
            "memory_enhanced": True
        }
    
    def handle_human_validation(state: AgentState) -> Dict:
        """Prepare for human validation if needed."""
        if state.get("needs_human_validation", False):
            return {"agent_state": state, "next": "human_validation", "agent": "HUMAN_VALIDATION"}
        return {"agent_state": state, "next": END}
    
    def perform_human_validation(state: AgentState) -> AgentState:
        """Handle human validation process."""
        print(f"Selected agent: HUMAN_VALIDATION")

        # Append validation request to the existing output
        validation_prompt = f"{state['output'].content}\n\n**Human Validation Required:**\n- If you're a healthcare professional: Please validate the output. Select **Yes** or **No**. If No, provide comments.\n- If you're a patient: Simply click Yes to confirm."

        # Create an AI message with the validation prompt
        validation_message = AIMessage(content=validation_prompt)

        return {
            **state,
            "output": validation_message,
            "agent_name": f"{state['agent_name']}, HUMAN_VALIDATION"
        }

    # Check output through guardrails
    def apply_output_guardrails(state: AgentState) -> AgentState:
        """Apply output guardrails to the generated response."""
        output = state["output"]
        current_input = state["current_input"]

        # Check if output is valid
        if not output or not isinstance(output, (str, AIMessage)):
            return state

        output_text = output if isinstance(output, str) else output.content
        
        # If the last message was a human validation message
        if "Human Validation Required" in output_text:
            # Check if the current input is a human validation response
            validation_input = ""
            if isinstance(current_input, str):
                validation_input = current_input
            elif isinstance(current_input, dict):
                validation_input = current_input.get("text", "")
            
            # If validation input exists
            if validation_input.lower().startswith(('yes', 'no')):
                # Add the validation result to the conversation history
                validation_response = HumanMessage(content=f"Validation Result: {validation_input}")
                
                # If validation is 'No', modify the output
                if validation_input.lower().startswith('no'):
                    fallback_message = AIMessage(content="The previous medical analysis requires further review. A healthcare professional has flagged potential inaccuracies.")
                    return {
                        **state,
                        "messages": [validation_response, fallback_message],
                        "output": fallback_message
                    }
                
                return {
                    **state,
                    "messages": validation_response
                }
        
        # Get the original input text
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Apply output sanitization
        sanitized_output = guardrails.check_output(output_text, input_text)
        # sanitized_output = output_text
        
        # For non-validation cases, add the sanitized output to messages
        sanitized_message = AIMessage(content=sanitized_output) if isinstance(output, AIMessage) else sanitized_output
        
        return {
            **state,
            "messages": sanitized_message,
            "output": sanitized_message
        }

    
    # Create the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes for each step
    workflow.add_node("analyze_input", analyze_input)
    workflow.add_node("route_to_agent", route_to_agent)
    workflow.add_node("CONVERSATION_AGENT", run_conversation_agent)
    workflow.add_node("RAG_AGENT", run_rag_agent)
    workflow.add_node("WEB_SEARCH_PROCESSOR_AGENT", run_web_search_processor_agent)
    workflow.add_node("MEDICAL_RESEARCH_AGENT", run_medical_research_agent)
    workflow.add_node("BRAIN_TUMOR_AGENT", run_brain_tumor_agent)
    workflow.add_node("CHEST_XRAY_AGENT", run_chest_xray_agent)
    workflow.add_node("SKIN_LESION_AGENT", run_skin_lesion_agent)
    workflow.add_node("check_validation", handle_human_validation)
    workflow.add_node("human_validation", perform_human_validation)
    workflow.add_node("apply_guardrails", apply_output_guardrails)
    
    # Define the edges (workflow connections)
    workflow.set_entry_point("analyze_input")
    # workflow.add_edge("analyze_input", "route_to_agent")
    # Add conditional routing for guardrails bypass
    workflow.add_conditional_edges(
        "analyze_input",
        check_if_bypassing,
        {
            "apply_guardrails": "apply_guardrails",
            "route_to_agent": "route_to_agent"
        }
    )
    
    # Connect decision router to agents
    workflow.add_conditional_edges(
        "route_to_agent",
        lambda x: x["next"],
        {
            "CONVERSATION_AGENT": "CONVERSATION_AGENT",
            "RAG_AGENT": "RAG_AGENT",
            "WEB_SEARCH_PROCESSOR_AGENT": "WEB_SEARCH_PROCESSOR_AGENT",
            "MEDICAL_RESEARCH_AGENT": "MEDICAL_RESEARCH_AGENT",
            "BRAIN_TUMOR_AGENT": "BRAIN_TUMOR_AGENT",
            "CHEST_XRAY_AGENT": "CHEST_XRAY_AGENT",
            "SKIN_LESION_AGENT": "SKIN_LESION_AGENT",
            "needs_validation": "RAG_AGENT"  # Default to RAG if confidence is low
        }
    )
    
    # Connect agent outputs to validation check
    workflow.add_edge("CONVERSATION_AGENT", "check_validation")
    # workflow.add_edge("RAG_AGENT", "check_validation")
    workflow.add_edge("WEB_SEARCH_PROCESSOR_AGENT", "check_validation")
    workflow.add_edge("MEDICAL_RESEARCH_AGENT", "check_validation")
    workflow.add_conditional_edges("RAG_AGENT", confidence_based_routing)
    workflow.add_edge("BRAIN_TUMOR_AGENT", "check_validation")
    workflow.add_edge("CHEST_XRAY_AGENT", "check_validation")
    workflow.add_edge("SKIN_LESION_AGENT", "check_validation")

    workflow.add_edge("human_validation", "apply_guardrails")
    workflow.add_edge("apply_guardrails", END)
    
    workflow.add_conditional_edges(
        "check_validation",
        lambda x: x["next"],
        {
            "human_validation": "human_validation",
            END: "apply_guardrails"  # Route to guardrails instead of END
        }
    )
    
    # workflow.add_edge("human_validation", END)
    
    # Compile the graph
    return workflow.compile(checkpointer=memory)


def init_agent_state() -> AgentState:
    """Initialize the agent state with default values."""
    return {
        "messages": [],
        "agent_name": None,
        "current_input": None,
        "has_image": False,
        "image_type": None,
        "output": None,
        "needs_human_validation": False,
        "retrieval_confidence": 0.0,
        "bypass_routing": False,
        "insufficient_info": False,
        "user_id": None,
        "memory_enhanced": False
    }


def process_query(query: Union[str, Dict], conversation_history: List[BaseMessage] = None, session_id: str = None) -> str:
    """
    Process a user query through the agent decision system.
    
    Args:
        query: User input (text string or dict with text and image)
        conversation_history: Optional list of previous messages, NOT NEEDED ANYMORE since the state saves the conversation history now
        session_id: Session ID for memory tracking and thread management
        
    Returns:
        Response from the appropriate agent
    """
    # Initialize the graph
    graph = create_agent_graph()

    # # # Save Graph Flowchart
    # image_bytes = graph.get_graph().draw_mermaid_png()
    # decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    # cv2.imwrite("./assets/graph.png", decoded)
    # print("Graph flowchart saved in assets.")
    
    # Initialize state
    state = init_agent_state()
    # if conversation_history:
    #     state["messages"] = conversation_history
    
    # Set user_id for memory tracking
    if session_id:
        state["user_id"] = session_id
    
    # Add the current query
    state["current_input"] = query

    # To handle image upload case
    if isinstance(query, dict):
        query = query.get("text", "") + ", user uploaded an image for diagnosis."
    
    state["messages"] = [HumanMessage(content=query)]

    # Use session-specific thread config for proper conversation history separation
    session_thread_config = {"configurable": {"thread_id": session_id or "default_thread"}}
    result = graph.invoke(state, session_thread_config)
    # print("######### DEBUG 4:", result)
    # state["messages"] = [result["messages"][-1].content]

    # Keep history to reasonable size (ANOTHER OPTION: summarize and store before truncating history)
    if len(result["messages"]) > config.max_conversation_history:  # Keep last config.max_conversation_history messages
        result["messages"] = result["messages"][-config.max_conversation_history:]

    # visualize conversation history in console
    for m in result["messages"]:
        m.pretty_print()
    
    # Add the response to conversation history
    return result
# Long-term Memory System for Multi-Agent Medical Assistant

This module implements intelligent long-term memory capabilities using mem0, enabling the medical assistant to learn from interactions, maintain user context, and provide personalized responses.

## Features

### 🧠 Memory Types
- **User Profile**: Personal information, communication preferences
- **Medical History**: Health-related queries, symptoms, and insights
- **Conversation Context**: Important discussion threads and follow-ups
- **Agent Learning**: Performance insights and user feedback
- **Medical Insights**: Clinical information and diagnostic results
- **Preferences**: User-specific settings and preferred interaction styles

### 🎯 Key Capabilities
- **Intelligent Storage**: Automatically categorizes and stores relevant information
- **Context-aware Retrieval**: Finds relevant memories based on current conversation
- **Prompt Enhancement**: Enriches agent prompts with personalized context
- **Medical Insight Tracking**: Maintains continuity across medical consultations
- **User Preference Learning**: Adapts communication style over time

## Setup Instructions

### 1. Install Dependencies
```bash
pip install mem0ai==0.1.32 redis==5.2.1
```

### 2. Configure Environment Variables
Copy the settings from `.env.memory.example` to your main `.env` file:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key

# Qdrant (for vector storage)
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Optional: Neo4j (for advanced graph relationships)
NEO4J_URL=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

### 3. Start Required Services

#### Option A: Using Docker (Recommended)
```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Optional: Start Neo4j
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/yourpassword neo4j:latest
```

#### Option B: Local Installation
Follow the installation guides for:
- [Qdrant](https://qdrant.tech/documentation/quick-start/)
- [Neo4j](https://neo4j.com/docs/operations-manual/current/installation/) (optional)

### 4. Test the System
```bash
cd memory/
python test_memory_system.py
```

## Usage

The memory system is automatically integrated into all agents in the `agent_decision.py` file. Each agent now:

1. **Retrieves relevant memories** before processing queries
2. **Enhances prompts** with personalized context
3. **Stores interactions** for future reference
4. **Tracks medical insights** across sessions

### Memory Enhancement per Agent

#### Conversation Agent
- Retrieves user preferences and communication style
- Maintains context across conversation threads
- Learns from interaction patterns

#### RAG Agent  
- Stores medical queries and responses
- Builds medical knowledge profile per user
- Provides consistent medical information

#### Medical Research Agent
- Tracks research topics and findings
- Avoids duplicate research requests
- Builds cumulative knowledge base

#### Image Analysis Agents
- Stores diagnostic results and image analysis history
- Maintains medical imaging timeline
- Tracks diagnostic patterns

## Configuration

Memory settings are configured in `config.py` under the `MemoryConfig` class:

```python
class MemoryConfig:
    def __init__(self):
        # Basic settings
        self.collection_name = "medical_assistant_longterm_memory"
        self.max_memories_per_query = 5
        self.memory_retention_days = 90
        
        # Enhancement settings
        self.enhance_conversation_prompts = True
        self.enhance_rag_prompts = True
        self.enhance_research_prompts = True
        self.max_context_length = 1000
```

## Memory Data Structure

Each memory entry contains:
- **Content**: The actual memory text
- **Metadata**: Structured information about the memory
- **User ID**: For user-specific storage
- **Timestamp**: When the memory was created
- **Agent Name**: Which agent created the memory
- **Memory Type**: Category of the memory
- **Confidence**: Reliability score

## Privacy and Security

- **User Isolation**: Memories are strictly separated by user ID
- **Data Retention**: Automatic cleanup of old conversation memories
- **Medical Compliance**: Special handling for medical information
- **Backup Support**: Regular backup capabilities for data protection

## Troubleshooting

### Common Issues

1. **Memory system fails to initialize**
   - Check if Qdrant is running on the configured port
   - Verify OpenAI API key is set correctly
   - Ensure all dependencies are installed

2. **Memory enhancement not working**
   - Check configuration flags in `MemoryConfig`
   - Verify memory system initialized successfully in logs
   - Run the test script to validate functionality

3. **Performance issues**
   - Reduce `max_memories_per_query` in config
   - Enable auto-cleanup to remove old memories
   - Consider increasing memory retention settings

### Logs to Monitor
```
✅ Long-term memory system initialized
✅ Enhanced [agent] prompt with long-term memory
✅ Stored [interaction] in long-term memory
⚠️ Memory enhancement failed, using base prompt
```

## Development

### Adding New Memory Types
1. Add the new type to `MemoryType` enum in `long_term_memory.py`
2. Create storage method in `LongTermMemoryManager`
3. Add retrieval logic if needed
4. Update prompt enhancement in `MemoryEnhancedPromptBuilder`

### Custom Memory Queries
Use the memory manager directly for custom operations:
```python
from memory import LongTermMemoryManager

memory_manager = LongTermMemoryManager(config)
memories = memory_manager.retrieve_relevant_memories(
    user_id="user123",
    query="diabetes management",
    limit=5
)
```

## Integration Status

✅ **Conversation Agent** - Enhanced prompts with user context  
✅ **RAG Agent** - Medical knowledge continuity  
✅ **Medical Research Agent** - Research history tracking  
✅ **Web Search Agent** - Search context enhancement  
✅ **Image Analysis Agents** - Diagnostic result storage  
✅ **Guardrails Integration** - Memory-aware safety checks 
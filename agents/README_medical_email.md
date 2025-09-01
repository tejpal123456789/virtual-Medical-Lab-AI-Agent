# Medical Research Email Assistant

This system combines the web research capabilities from `bright_data.py` with the Gmail sending functionality from `tool_arcade.py` to create a medical research assistant that can email results.

## Files Created

1. **`medical_research_emailer.py`** - Main script (simple, direct)
2. **`combined_research_email.py`** - Full-featured version 
3. **`demo_medical_email.py`** - Demo examples

## Quick Start

### 1. Environment Setup
Make sure your `.env` file has:
```
OPENAI_API_KEY=your_openai_key
BRIGHT_DATA_API_TOKEN=your_bright_data_token
ARCADE_API_KEY=your_arcade_key
ARCADE_USER_ID=your_arcade_user_id
```

### 2. Run the Simple Version
```bash
cd agents
python medical_research_emailer.py
```

### 3. Example Usage
The script will:
1. Ask for a medical research topic (e.g., "COVID-19 latest treatments")
2. Use Bright Data tools to research current medical information
3. Ask for recipient email address
4. Send research summary via Gmail using Arcade authentication

## Features

- ✅ Web scraping for current medical research
- ✅ Gmail integration with Arcade authentication
- ✅ Human-in-the-loop approval for email sending
- ✅ Concise research summaries optimized for email
- ✅ Professional email formatting

## How It Works

1. **Research Phase**: Uses Bright Data MCP tools to scrape medical websites, journals, and news sources
2. **Summarization**: AI agent creates concise medical research summary (<1500 chars)
3. **Email Phase**: Uses Arcade Gmail_SendEmail tool with proper authentication
4. **Security**: Human approval required before sending emails

## Example Research Topics

- "Type 2 Diabetes treatment guidelines 2024"
- "Cancer immunotherapy latest breakthroughs"
- "Mental health telemedicine effectiveness"
- "Alzheimer's disease recent clinical trials"

## Troubleshooting

If you get authentication errors:
1. Check that all environment variables are set correctly
2. Make sure you've authorized the Gmail tool in Arcade
3. Verify your Arcade user ID is correct 
# Gen-AI


#  LangGraph Multi-Agent AI System

A powerful multi-agent system built using [LangGraph](https://github.com/langchain-ai/langgraph), designed to handle complex queries using specialized agents:

-  **Web Search Agent** — Fetches real-time web results via SerpAPI.
-  **YouTube Agent** — Extracts transcripts, metadata, and summaries from YouTube videos.
-  **Wikipedia Agent** — Pulls structured data from Wikipedia articles.
-  **Groq LLM Chatbot Agent** — Answers general questions using Groq's LLM.

##  Features

- **Dynamic task routing** using a LangGraph router node
- Modular agents with clearly defined responsibilities
- Real-time information retrieval
- Integration with external APIs (SerpAPI, YouTube Transcript API)
- Structured responses from all agents

## Project Structure
multi-agent-system/ ├── agents/ │ ├── chatbot_agent.py │ ├── web_search_agent.py │ ├── youtube_agent.py │ ├── wikipedia_agent.py │ └── router.py ├── main.py # LangGraph app setup ├── utils.py # Helper functions ├── requirements.txt └── README.md

## Example Query
User: Can you summarize this YouTube video? https://www.youtube.com/watch?v=dQw4w9WgXcQ
Response: **Title**: Rick Astley - Never Gonna Give You Up  
**Channel**: Rick Astley  
**Summary**: This is the iconic 1987 music video of Rick Astley's hit single. The lyrics express loyalty and devotion, promising never to give up on someone.  
**Transcript**: [Full transcript here...]

(Video URL: https://www.youtube.com/watch?v=dQw4w9WgXcQ)

Query:
User: What’s the weather like in Tokyo right now?
Response (from Web Search Agent via SerpAPI):
Currently, it's 21°C and sunny in Tokyo, Japan.

Query:
User: What's the difference between supervised and unsupervised learning?
Response (from Groq LLM):
Supervised learning uses labeled data to train models, while unsupervised learning finds patterns

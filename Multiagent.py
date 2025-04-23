from langgraph.graph import StateGraph
from langchain_community.utilities import WikipediaAPIWrapper, SerpAPIWrapper
from langchain_groq import ChatGroq
from youtube_transcript_api import YouTubeTranscriptApi
from pydantic import BaseModel
from dotenv import load_dotenv
import os, re

# Load keys
load_dotenv()
GROQ_KEY = os.getenv("GROQ_KEYS")
SERPAPI_KEY = os.getenv("SERPAPI_KEYS")

# Initialize tools
groq = ChatGroq(api_key=GROQ_KEY, model_name="llama-3.3-70b-versatile")
search = SerpAPIWrapper(serpapi_api_key=SERPAPI_KEY)
wiki = WikipediaAPIWrapper()

# Shared state
class AgentState(BaseModel):
    query: str
    source: str = ""
    response: str = ""
    extra: str = ""

# Helper for YouTube
def extract_video_id(url):
    match = re.search(r"(?:v=|youtu.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else ""

# Router
def router(state: AgentState) -> str:
    query = state.query.lower()
    if "video" in query or "youtube" in query:
        return "youtube_agent"
    elif "wikipedia" in query or "who is" in query:
        return "wikipedia_agent"
    elif "search" in query or "news" in query or "latest" in query or 'today' in query:
        return "web_search_agent"
    else:
        return "groq_agent"

# Groq LLM agent
def groq_agent(state: AgentState) -> dict:
    answer = groq.invoke(state.query)
    return {
        "query": state.query,
        "source": "Groq LLM",
        "response": str(answer.content),
        "extra": ""
    }


# Wikipedia agent
def wikipedia_agent(state: AgentState) -> dict:
    answer = wiki.run(state.query)
    return {
        "query": state.query,
        "source": "Wikipedia",
        "response": str(answer),
        "extra": ""
    }


# Web Search agent
def web_search_agent(state: AgentState) -> dict:
    try:
        results = search.results(state.query)
        answer = ""

        # Try to extract from organic results
        organic = results.get("organic_results", [])
        if organic:
            for result in organic[:3]:  # Limit to top 3
                title = result.get("title", "No title")
                link = result.get("link", "")
                snippet = result.get("snippet", "")
                answer += f"🔹 {title}\n🔗 {link}\n📄 {snippet}\n\n"
        else:
            answer = "No results found."

    except Exception as e:
        answer = f"Search error: {e}"

    return {
        "query": state.query,
        "source": "Web Search",
        "response": answer.strip(),
        "extra": ""
    }



def youtube_agent(state: AgentState) -> dict:
    results = search.results(f"{state.query} site:youtube.com")
    video = results.get("organic_results", [])[0]
    url = video.get("link", "")
    title = video.get("title", "")
    video_id = extract_video_id(url)

    transcript = "Transcript not available."
    summary = "Summary not available."

    try:
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([chunk["text"] for chunk in transcript_data])
        
        # Generate summary using Groq
        summary_prompt = f"Summarize this YouTube transcript:\n\n{transcript[:4000]}"
        summary_response = groq.invoke(summary_prompt)
        summary = summary_response.content if hasattr(summary_response, "content") else str(summary_response)

        # Shorten transcript to preview (optional)
        transcript = transcript[:300] + "..."

    except Exception as e:
        print(f"[Transcript error] {e}")

    return {
        "query": state.query,
        "source": "YouTube",
        "response": (
            f"🎥 Title: {title}\n"
            f"🔗 Link: {url}\n"
            f"📄 Transcript: {transcript}\n"
            f"📝 Summary: {summary}"
        ),
        "extra": url
    }

# Build LangGraph
graph = StateGraph(AgentState)
graph.add_node("router", lambda state: state)  # router doesn't modify state
graph.add_node("groq_agent", groq_agent)
graph.add_node("wikipedia_agent", wikipedia_agent)
graph.add_node("web_search_agent", web_search_agent)
graph.add_node("youtube_agent", youtube_agent)

# Router routing logic
graph.add_conditional_edges("router", router, {
    "groq_agent": "groq_agent",
    "wikipedia_agent": "wikipedia_agent",
    "web_search_agent": "web_search_agent",
    "youtube_agent": "youtube_agent",
})

# Terminal states
graph.set_entry_point("router")
graph.set_finish_point("groq_agent")
graph.set_finish_point("wikipedia_agent")
graph.set_finish_point("web_search_agent")
graph.set_finish_point("youtube_agent")

# Compile graph
app = graph.compile()

# Run interactively
while True:
    user_input = input("\n🧠 Ask me anything (type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    result = app.invoke({"query": user_input})
    print(f"\n🔎 Source: ",result['source'])
    print(f"\n🧠 Answer:\n", result['response'])
    if result['extra']:
      print(f"\n📎 Extra:\n", result['extra'])


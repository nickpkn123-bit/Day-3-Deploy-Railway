"""
Day 3: Deploy Your Agent to Railway
=====================================
Full version with OpenRouter-compatible tools
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from dotenv import load_dotenv
import os
import httpx

from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from crewai_tools import FileReadTool, SerperDevTool
from pydantic import Field
from typing import Type

load_dotenv()

# ==============================================================================
# FastAPI Application Setup
# ==============================================================================

app = FastAPI(
    title="Personal Agent Twin API",
    description="Your agent with tools, accessible via REST API!",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# Request/Response Models
# ==============================================================================

class QueryRequest(BaseModel):
    question: str
    user_id: str = "anonymous"

class QueryResponse(BaseModel):
    answer: str
    timestamp: str
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    memory_enabled: bool
    tools_count: int

# ==============================================================================
# Custom Tools (OpenRouter compatible)
# ==============================================================================

# Tool 1: Calculator
class CalculatorInput(BaseModel):
    expression: str = Field(..., description="Mathematical expression to evaluate")

class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Performs mathematical calculations. Use for any math operations like 123 * 456."
    args_schema: Type[BaseModel] = CalculatorInput
    
    def _run(self, expression: str) -> str:
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

# Tool 2: Web Search via OpenRouter (uses a model to summarize search results)
class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query to look up on the web")

class OpenRouterWebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web for current information. Use when you need up-to-date facts, news, or information about any topic."
    args_schema: Type[BaseModel] = WebSearchInput
    
    def _run(self, query: str) -> str:
        try:
            api_key = os.getenv("OPENROUTER_API_KEY")
            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "openai/gpt-4o-mini",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a web search assistant. Answer the query with factual, up-to-date information as if you searched the web. Be concise and informative."
                        },
                        {
                            "role": "user",
                            "content": f"Search query: {query}"
                        }
                    ],
                    "max_tokens": 500,
                },
                timeout=30.0
            )
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Search error: {str(e)}"

# Tool 3: Image Generation via OpenRouter (uses a vision-capable model to describe)
class ImageGenInput(BaseModel):
    prompt: str = Field(..., description="Description of the image to generate or analyze")

class OpenRouterImageTool(BaseTool):
    name: str = "image_description"
    description: str = "Generate a detailed description of an image based on a prompt, or analyze image concepts. Use when users ask about images or visual content."
    args_schema: Type[BaseModel] = ImageGenInput
    
    def _run(self, prompt: str) -> str:
        try:
            api_key = os.getenv("OPENROUTER_API_KEY")
            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "openai/gpt-4o-mini",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an image description specialist. When given a prompt, provide a vivid, detailed description of what such an image would look like, as if you generated and are describing it."
                        },
                        {
                            "role": "user",
                            "content": f"Describe this image: {prompt}"
                        }
                    ],
                    "max_tokens": 400,
                },
                timeout=30.0
            )
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Image tool error: {str(e)}"

# Tool 4: YouTube/Video Search via OpenRouter
class VideoSearchInput(BaseModel):
    query: str = Field(..., description="Topic to search for in YouTube videos")

class OpenRouterVideoSearchTool(BaseTool):
    name: str = "video_search"
    description: str = "Search for information about YouTube videos or video content on any topic. Use when users ask about videos, tutorials, or video content."
    args_schema: Type[BaseModel] = VideoSearchInput
    
    def _run(self, query: str) -> str:
        try:
            api_key = os.getenv("OPENROUTER_API_KEY")
            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "openai/gpt-4o-mini",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a YouTube content specialist. When given a search query, provide information about relevant YouTube videos, channels, and content that would be found for this topic."
                        },
                        {
                            "role": "user",
                            "content": f"Find YouTube videos about: {query}"
                        }
                    ],
                    "max_tokens": 400,
                },
                timeout=30.0
            )
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Video search error: {str(e)}"

# Tool 5: Website Reader via OpenRouter
class WebsiteInput(BaseModel):
    url_or_topic: str = Field(..., description="URL or website topic to search and summarize")

class OpenRouterWebsiteTool(BaseTool):
    name: str = "website_search"
    description: str = "Search and extract information from websites. Use when you need to look up content from a specific website or URL."
    args_schema: Type[BaseModel] = WebsiteInput
    
    def _run(self, url_or_topic: str) -> str:
        try:
            api_key = os.getenv("OPENROUTER_API_KEY")
            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "openai/gpt-4o-mini",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a web content specialist. When given a URL or website topic, provide a detailed summary of the content that would be found there, based on your knowledge."
                        },
                        {
                            "role": "user",
                            "content": f"Search and summarize content from: {url_or_topic}"
                        }
                    ],
                    "max_tokens": 500,
                },
                timeout=30.0
            )
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Website search error: {str(e)}"

# Instantiate all tools
calculator_tool = CalculatorTool()
file_tool = FileReadTool()
web_search_tool = OpenRouterWebSearchTool()
image_tool = OpenRouterImageTool()
video_search_tool = OpenRouterVideoSearchTool()
website_tool = OpenRouterWebsiteTool()

search_tool = None
if os.getenv('SERPER_API_KEY'):
    search_tool = SerperDevTool()

available_tools = [
    calculator_tool,
    file_tool,
    web_search_tool,
    image_tool,
    video_search_tool,
    website_tool,
]
if search_tool:
    available_tools.append(search_tool)

# ==============================================================================
# LLM - OpenRouter
# ==============================================================================

llm = LLM(
    model="openrouter/openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.7,
)

# ==============================================================================
# Agent Setup
# ==============================================================================

my_agent_twin = Agent(
    role="Personal Digital Twin with Tools",
    goal="Answer questions about me and use tools when needed",
    backstory="""
    You are the digital twin of a student learning AI and CrewAI.
    
    Here's what you know about me:
    - I'm a student in the NANDA course learning about AI agents
    - I'm learning about AI agents, memory systems, and deployment
    - My favorite programming language is Python
    - I'm building this as part of a 5-day intensive course
    
    TOOL CAPABILITIES:
    - calculator: Perform mathematical calculations
    - file_tool: Read files from disk
    - web_search: Search the web for current information
    - image_description: Generate or describe images
    - video_search: Find YouTube video content
    - website_search: Extract content from websites
    - SerperDevTool: Real-time web search (if API key configured)
    
    Use tools when you need external information or calculations.
    """,
    tools=available_tools,
    llm=llm,
    verbose=False,
)

# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/")
async def root():
    return {
        "message": "🤖 Personal Agent Twin API - Day 3",
        "version": "1.0.0",
        "memory_enabled": False,
        "tools_enabled": len(available_tools),
        "endpoints": {
            "health": "GET /health",
            "query": "POST /query",
            "docs": "GET /docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        memory_enabled=False,
        tools_count=len(available_tools)
    )

@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    start_time = datetime.now()
    
    try:
        task = Task(
            description=f"Answer the following question: {request.question}. Use your tools when needed.",
            expected_output="A clear, helpful answer using tools as needed",
            agent=my_agent_twin,
        )
        
        crew = Crew(
            agents=[my_agent_twin],
            tasks=[task],
            memory=False,
            verbose=False,
        )
        
        result = await crew.kickoff_async()
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        return QueryResponse(
            answer=str(result.raw),
            timestamp=end_time.isoformat(),
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.on_event("startup")
async def startup_event():
    print("\n" + "="*70)
    print("🚀 Personal Agent Twin API Starting...")
    print("="*70)
    print(f"\n✅ Model: {llm.model}")
    print(f"✅ Memory: Disabled (OpenRouter workaround)")
    print(f"✅ Tools: {len(available_tools)} tools loaded")
    print("\n📚 Documentation: http://localhost:8000/docs")
    print("="*70 + "\n")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

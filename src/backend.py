from pydantic import BaseModel
from typing import List
from fastapi import FastAPI
from src.ai_agent import get_response_from_ai_agent
from langsmith import traceable
import uvicorn

#Step1: Setup Pydantic Model (Schema Validation)
class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool

#Step2: Setup AI Agent from FrontEnd Request
app = FastAPI(title="AI Agent API")
ALLOWED_MODEL_NAMES=["llama3-70b-8192", "mixtral-8x7b-32768", "llama-3.3-70b-versatile", "gpt-4o-mini"]

@app.post("/chat")
@traceable
def chat_endpoint(request_state: RequestState):
    """
    API Endpoint to interact with the Chatbot using LangGraph and search tools.
    It dynamically selects the model specified in the request
    """
    if request_state.model_name not in ALLOWED_MODEL_NAMES:
        return {"error": "Invalid model name"}
    
    llm_id = request_state.model_name
    query = request_state.messages
    allow_search = request_state.allow_search
    system_prompt = request_state.system_prompt
    provider = request_state.model_provider

    response = get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider)
    return response

#Step3: Run app & Explore Swagger UI Docs
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)
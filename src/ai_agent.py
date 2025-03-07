import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage
from src.logger import logging
from src.exception import CustomException
from langsmith import traceable
import sys

load_dotenv()


# Step 1: Load API Key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")

# Enable LangSmith Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

if not GROQ_API_KEY or not TAVILY_API_KEY or not OPENAI_API_KEY:
    logging.error("API keys are missing. Please check your environment variables.")
    raise ValueError("API keys are missing. Please check your environment variables.")

openai_llm=ChatOpenAI(model="gpt-4o-mini")
groq_llm=ChatGroq(model="llama-3.3-70b-versatile")


def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    logging.info("Getting query from AI agent")
    try:
        if provider == "groq":
            llm = ChatGroq(model=llm_id)
        elif provider == "openai":
            llm = ChatOpenAI(model=llm_id)
        system_prompt = "Act as an AI chatbot who is smart and friendly"
        tools=[TavilySearchResults(max_results=2)] if allow_search else []
        agent=create_react_agent(
            model=llm,
            tools=tools,
            state_modifier=system_prompt
        )
        state={"messages": query}
        response=agent.invoke(state)
        messages = response.get("messages")
        ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]
        logging.info("Response generated sucessfully")
        return ai_messages[-1]
    except Exception as e:
        raise CustomException(e, sys)
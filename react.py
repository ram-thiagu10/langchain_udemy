from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os
load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature = 0
)

@tool
def triple(num: float)-> float:
    """
    param num: a number to triple
    returns: the triple of the input number
    """
    return float(num) * 3


tools = [TavilySearch(max_result=1), triple]
llm = llm.bind_tools(tools)


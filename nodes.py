from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from react import llm, tools
load_dotenv()

SYSTEM_MESSAGE = """
You are a helpful assistant that can use tool to answer question
"""

def run_agent_reasoning(state:MessagesState) -> MessagesState:
    """
    Run the agent reasoning node
    """
    response = llm.invoke([{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]])
    return {"message": [response]}

tool_node = ToolNode(tools)


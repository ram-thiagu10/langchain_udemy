from dotenv import load_dotenv
import os
load_dotenv()  # Load environment variables from .env file
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
MAX_ITERATIONS = 8
# MODEL = "qwen3.5:0.8b"
MODEL = "llama-3.3-70b-versatile"

@tool
def get_product_price(product:str) -> float:
    """Look up the price of a product in catalog and return the price."""
    print(f"Executing get_product_price(product = '{product}')")
    prices = {"laptop": 45000.0, "headphones": 2000.0, "keyboard": 1500.0, "mouse": 800.0}
    return prices.get(product.lower(), 0.0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to the price based on the discount tier.
    Available tiers are silver, gold, platinum"""
    print(f"Executing apply_discount(price = {price}, discount_tier = '{discount_tier}')")
    discounts = {"silver": 0.05, "gold": 0.10, "platinum": 0.15}
    discount_rate = discounts.get(discount_tier.lower(), 0.0)
    return round(price * (1 - discount_rate), 2)

@traceable(name = "Langchain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    # llm = init_chat_model(f"ollama: {MODEL}", temperature=0.0)
    llm = init_chat_model(f"groq:{MODEL}", temperature=0.0)
    llm_with_tools = llm.bind_tools(tools)
    print(f"Question : {question}")
    print("="*100)

    messages = [
        SystemMessage(
            content = "You are a helpful shopping assistant"
            "You have access to product catalog"
            "and a discount tools.\n\n"
            "STRICT RULES - You must follow these exactly:\n"
            "1. Never guess or assume any product price.\n"
            "2. Always use the get_product_price tool to look up product prices.\n"
            "3. Always use the apply_discount tool to apply discounts after you have received"
            "a price from get_product_price tool. Pass the exact price returned by get_product_price "
            "- do NOT pass a madeup number.\n"
            "4. If the user does not specify a discount tier, ask them which tier to use. Do not assume one."
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"Iteration {iteration} of {MAX_ITERATIONS}")
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        #if no tool_calls, this is the final answer
        if not tool_calls:
            print(f"Final Answer is : {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        
        print(f"[Tool Selected] {tool_name} with args: {tool_args}")
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found in tools_dict")

        observation = tool_to_use.invoke(tool_args)
        print(f"Tool Result - {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content = str(observation), tool_call_id = tool_call_id)
        )

    print("Error - MAX iteration reached without final answer")
    return None





if __name__ == "__main__":
    print("Hello langchain agent (.bind_tools)")
    print()
    question = "What is the price of a keyboard after applying a gold discount?"
    result = run_agent(question)
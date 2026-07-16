from dotenv import load_dotenv
import os
load_dotenv()  # Load environment variables from .env file
import ollama
import re, inspect
from langsmith import traceable
MAX_ITERATIONS = 8
MODEL = "qwen3:1.7b"
# MODEL = "llama-3.3-70b-versatile"

@traceable(run_type = "tool")
def get_product_price(product:str) -> float:
    """Look up the price of a product in catalog and return the price."""
    print(f"Executing get_product_price(product = '{product}')")
    prices = {"laptop": 45000.0, "headphones": 2000.0, "keyboard": 500.0, "mouse": 200.0}
    return prices.get(product.lower(), 0.0)

@traceable(run_type = "tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to the price based on the discount tier.
    Available tiers are silver, gold, platinum"""
    print(f"Executing apply_discount(price = {price}, discount_tier = '{discount_tier}')")
    discounts = {"silver": 0.10, "gold": 0.15, "platinum": 0.20}
    discount_rate = discounts.get(discount_tier.lower(), 0.0)
    price = float(price)
    return round(price * (1 - discount_rate), 2)

tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}

def get_tool_description(tool_dict):
    description = []
    for tool_name, tool_function in tool_dict.items():
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        description.append(f"{tool_name}{signature}-{docstring}")
    return "\n".join(description)


tool_descriptions = get_tool_description(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""

STRICT RULES - You must follow these exactly:
1. Never guess or assume any product price.
2. Always use the get_product_price tool to look up product prices.
3. Always use the apply_discount tool to apply discounts after you have received a price from get_product_price tool. Pass the exact price returned by get_product_price - do NOT pass a madeup number.
4. If the user does not specify a discount tier, ask them which tier to use. Do not assume one.
5. Never assume any product name Strictly get the product name from the user and pass it to get_product_price tool.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:
"""


@traceable(name = "Ollama chat", run_type = "llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model = model, messages = messages, options = options)

@traceable(name = "Ollama Agent Loop")
def run_agent(question: str):
    
    print(f"Question : {question}")
    print("="*100)

    prompt = react_prompt.format(question=question)
    scratchpad = ""

    for iteration in range(1, MAX_ITERATIONS+1):
        full_prompt = prompt + scratchpad
        print(f"Iteration {iteration} of {MAX_ITERATIONS}")
        response = ollama_chat_traced(
            model = MODEL,
            messages=[{"role":"user", "content": full_prompt}],
            options = {"stop": ["\nObservation"], "temperature": 0})
        output = response.message.content
        print(f"LLM Output - {output}")
        print(f"   [Parsing] Looking for final answer in LLM output...")
        final_answer_match = re.search(f"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"   [Parser] Final Answer - {final_answer}")
            print("\n"+"="*100)
            print(f"Final Answer - {final_answer}")
            return final_answer

        
        print(f"[Parsing] Looking for Action and Action input in LLM Output....")
        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)
        if not action_match or not action_input_match:
            print(f"[Parsing] Error - Could not find Action or Action Input in LLM output. Stopping.")
            break
        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()
        print(f"[Tool Selected] {tool_name} with args: {tool_input_raw}")

        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f" Tool Executing {tool_name}({args})...")
        if tool_name not in tools:
            observation = f"Error - Tool {tool_name} not found in tools_dict"
        else:
            observation = str(tools[tool_name](*args))
        print(f"  [Tool Result] {observation}")

        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    print("Error - MAX iteration reached without final answer")
    return None





if __name__ == "__main__":
    print("Hello langchain agent (.bind_tools)")
    print()
    question = "What is the price of a mouse after applying a gold discount?"
    result = run_agent(question)
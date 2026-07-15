from dotenv import load_dotenv
import os
load_dotenv()  # Load environment variables from .env file
import ollama

from langsmith import traceable
MAX_ITERATIONS = 8
MODEL = "qwen3.5:0.8b"
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
    return round(price * (1 - discount_rate), 2)

tools_for_llm = [
    {
      "type": "function",
      "function": {
        "name": "get_product_price",
        "description": "Get the price of a product from catalog. Available products are laptop, headphones, keyboard, mouse",
        "parameters": {
          "type": "object",
          "required": ["product"],
          "properties": {
            "product": {"type": "string", "description": "The name of the product"}
          }
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "apply_discount",
        "description": "Apply a discount to the price based on the discount tier. Available tiers are silver, gold, platinum",
        "parameters": {
          "type": "object",
          "required": ["price", "discount_tier"],
          "properties": {
            "price": {"type": "number", "description": "The price of the product"},
            "discount_tier": {"type": "string", "description": "The discount tier to apply"}
          }
        }
      }
    }
  ]

@traceable(name = "Ollama chat", run_type = "llm")
def ollama_chat_traced(messages):
    return ollama.chat(MODEL, messages, tools=tools_for_llm)

@traceable(name = "Ollama Agent Loop")
def run_agent(question: str):
    tools_dict = {
        "get_product_price" : get_product_price,
        "apply_discount" : apply_discount
    }
   
    print(f"Question : {question}")
    print("="*100)

    messages = [
        {"role":"system",
            "content":("You are a helpful shopping assistant"
            "You have access to product catalog"
            "and a discount tools.\n\n"
            "STRICT RULES - You must follow these exactly:\n"
            "1. Never guess or assume any product price.\n"
            "2. Always use the get_product_price tool to look up product prices.\n"
            "3. Always use the apply_discount tool to apply discounts after you have received"
            "a price from get_product_price tool. Pass the exact price returned by get_product_price "
            "- do NOT pass a madeup number.\n"
            "4. If the user does not specify a discount tier, ask them which tier to use. Do not assume one."
            "5. Never assume any product name Strictly get the product name from the user and pass it to get_product_price tool."
        )},
        {
            "role": "user",
            "content": question
        }
    ]

    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"Iteration {iteration} of {MAX_ITERATIONS}")
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message
        tool_calls = ai_message.tool_calls

        #if no tool_calls, this is the final answer
        if not tool_calls:
            print(f"Final Answer is : {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
        
        print(f"[Tool Selected] {tool_name} with args: {tool_args}")
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found in tools_dict")

        observation = tool_to_use(**tool_args)
        print(f"Tool Result - {observation}")

        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                "content": str(observation),
            }
        )

    print("Error - MAX iteration reached without final answer")
    return None





if __name__ == "__main__":
    print("Hello langchain agent (.bind_tools)")
    print()
    question = "What is the price of a keyboard after applying a gold discount?"
    result = run_agent(question)
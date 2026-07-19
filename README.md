# AI Agent Tool Calling - Three Approaches

This repository demonstrates **three different ways of building an AI Agent that can use tools**.

All three implementations solve the **same problem**:

> Given a user's question, the agent should determine which tools to call, execute them in the correct order, observe the results, and finally answer the user.

Although the functionality is identical, the implementation approach is completely different.

---

# Repository Structure

```
1_agent_loop_langchain_tool_calling.py
2_agent_loop_raw_function_calling.py
3_raw+react+prompt.py
README.md
```

---

# What is an AI Agent?

An AI Agent is an LLM that can:

- Understand a goal
- Decide what actions are required
- Choose the appropriate tool(s)
- Execute those tools
- Observe the result
- Continue reasoning until the task is completed

Instead of answering immediately,

```
Question
↓

Answer
```

an AI Agent follows an iterative reasoning loop:

```
Question
      ↓
Think
      ↓
Choose Tool
      ↓
Execute Tool
      ↓
Observe Result
      ↓
Think Again
      ↓
Repeat
      ↓
Final Answer
```

This iterative reasoning process is what makes the system "agentic".

---

# AI Agent vs Agentic AI vs Multi-Agent

These terms are often confused.

## AI Agent

One LLM that can use tools to accomplish a task.

```
          User
            │
            ▼
        AI Agent
        /      \
    Tool A    Tool B
```

All three implementations in this repository are **AI Agents**.

---

## Agentic AI

Agentic AI describes the **behavior** rather than the architecture.

An LLM becomes agentic when it can:

- plan
- act
- observe
- continue planning
- finish the task

Every implementation in this repository demonstrates **agentic behavior**.

---

## Multi-Agent

Multi-Agent systems contain **multiple independent AI Agents** collaborating.

Example:

```
                 Supervisor Agent
                 /      |      \
                ▼       ▼       ▼
          Search    Finance   Coding
            Agent     Agent     Agent
```

This repository **does not implement Multi-Agent systems**.

---

# The Three Implementations

---

# 1. LangChain Tool Calling

File:

```
1_agent_loop_langchain_tool_calling.py
```

## Overview

This implementation uses:

- LangChain
- `@tool`
- `bind_tools()`

LangChain automatically converts Python functions into tool schemas and handles communication with the model.

Instead of writing JSON schemas manually, developers only define Python functions.

Example:

```python
@tool
def get_product_price(product: str):
    ...
```

LangChain automatically creates

- tool name
- description
- parameters
- JSON schema

and sends them to the model.

---

## Flow

```
User
   │
   ▼
LangChain
   │
   ▼
LLM
   │
Tool Call
   │
   ▼
Python Tool
   │
Observation
   │
   ▼
LLM
   │
Final Answer
```

---

## Advantages

- Very little boilerplate
- Automatic schema generation
- Provider independent
- Supports OpenAI, Groq, Ollama, Anthropic etc.
- Easy integration with LangSmith
- Easy integration with LangGraph

---

## Disadvantages

- Less visibility into the internal tool calling protocol
- Extra dependency (LangChain)
- Additional abstraction layer

---

# 2. Raw Function Calling

File:

```
2_agent_loop_raw_function_calling.py
```

## Overview

This implementation directly uses the model provider's **native function calling API**.

You manually define tool schemas.

Example:

```python
tools_for_llm = [
    {
        "type":"function",
        ...
    }
]
```

The model returns structured tool calls.

Example:

```json
{
    "tool_calls":[
        {
            "name":"get_product_price",
            "arguments":{
                "product":"mouse"
            }
        }
    ]
}
```

No parsing is required.

---

## Flow

```
User
   │
   ▼
LLM
   │
Tool Call JSON
   │
   ▼
Python Tool
   │
Observation
   │
   ▼
LLM
   │
Final Answer
```

---

## Advantages

- Native implementation
- No framework dependency
- Structured tool calls
- Reliable
- Easy to debug
- Fast

---

## Disadvantages

- JSON schemas must be written manually
- Provider-specific implementation
- Different providers have slightly different APIs

---

# 3. Raw ReAct Prompting

File:

```
3_raw+react+prompt.py
```

## Overview

This implementation predates native function calling.

The model is instructed to produce text in the famous **ReAct format**.

Example:

```
Thought:
I need the product price.

Action:
get_product_price

Action Input:
mouse
```

Your Python code parses the generated text using regular expressions.

Example:

```python
action_match = re.search(...)
```

After executing the tool,

the observation is manually appended back into the prompt.

```
Observation:
200

Thought:
```

The model continues reasoning.

---

## Flow

```
User
   │
Prompt
   │
   ▼
LLM

Thought

Action

Action Input

   │
Regex Parser
   │
Python Tool
   │
Observation
   │
Prompt Again
   │
LLM
```

---

## Advantages

- Maximum flexibility
- Complete visibility
- Excellent for learning
- Works even with models that do not support tool calling

---

## Disadvantages

- Extremely fragile
- Depends on prompt formatting
- Requires regex parsing
- Easy to break
- More prompt engineering
- More maintenance

---

# Comparison

| Feature | LangChain Tool Calling | Raw Function Calling | Raw ReAct Prompt |
|----------|------------------------|----------------------|------------------|
| Framework | LangChain | None | None |
| Native Tool Calling | ✅ | ✅ | ❌ |
| Manual Prompt Engineering | Minimal | Minimal | Heavy |
| Regex Parsing | ❌ | ❌ | ✅ |
| JSON Schema Required | ❌ | ✅ | ❌ |
| Automatic Schema Generation | ✅ | ❌ | ❌ |
| Structured Tool Calls | ✅ | ✅ | ❌ |
| Reliability | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Learning Value | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Production Ready | ✅ | ✅ | ⚠️ |
| Provider Independent | ✅ | ❌ | Mostly |
| Boilerplate | Low | Medium | High |

---

# Developer Control Comparison

One of the biggest differences between these approaches is **how much control the developer has**.

## ReAct Prompt

```
Developer controls EVERYTHING
```

You control:

- Prompt format
- Thought format
- Action format
- Parsing
- Tool execution
- Observation formatting
- Iteration logic
- Error handling

Nothing is hidden.

This is the best approach to understand how AI Agents work internally.

However, it also means the developer is responsible for everything.

---

## Raw Function Calling

```
Developer controls almost everything
```

The model only decides:

- Which tool to call
- Tool arguments

The developer controls:

- Tool schemas
- Agent loop
- Tool execution
- Memory
- Error handling
- Retry logic
- Conversation management

This gives excellent balance between flexibility and reliability.

---

## LangChain

```
Developer delegates many responsibilities
```

LangChain manages:

- Schema generation
- Tool parsing
- Message objects
- Provider compatibility

The developer focuses on:

- Business logic
- Tool implementation
- Workflow

This reduces code significantly.

---

# Control Comparison

| Aspect | LangChain | Raw Function Calling | ReAct |
|----------|-----------|----------------------|--------|
| Prompt Control | Medium | High | Very High |
| Agent Loop Control | High | Very High | Very High |
| Tool Execution Control | High | Very High | Very High |
| Tool Schema Control | Low | Very High | N/A |
| Parsing Control | Hidden | High | Very High |
| Conversation Management | High | Very High | Very High |
| Debugging Visibility | Medium | High | Very High |
| Customization | High | Very High | Unlimited |

---

# Which Approach Should You Use?

## If your goal is learning AI Agents

Choose:

```
3_raw+react+prompt.py
```

Why?

Because you'll understand:

- ReAct prompting
- Agent loops
- Tool execution
- Scratchpads
- Observations
- Iterative reasoning

Every modern agent framework evolved from this idea.

---

## If you want production code without a framework

Choose:

```
2_agent_loop_raw_function_calling.py
```

Why?

- Reliable
- Native API
- Structured outputs
- Full developer control
- Less abstraction
- Easier debugging

Many production systems use this approach directly with OpenAI, Anthropic, Groq, Gemini, or Ollama.

---

## If you're already using LangChain

Choose:

```
1_agent_loop_langchain_tool_calling.py
```

Why?

- Less code
- Automatic tool schemas
- Easy provider switching
- Integrates with LangGraph
- Integrates with LangSmith
- Easier scaling

---

# Recommended Learning Order

If you're new to AI Agents, follow this order:

```
Step 1

3_raw+react+prompt.py

↓

Understand:
- ReAct
- Agent Loop
- Thought
- Action
- Observation

↓

Step 2

2_agent_loop_raw_function_calling.py

↓

Understand:
- Native Function Calling
- Tool Schemas
- Structured Outputs

↓

Step 3

1_agent_loop_langchain_tool_calling.py

↓

Understand:
- Framework Abstractions
- bind_tools()
- @tool
- ToolMessage
```

Following this progression makes it much easier to appreciate what frameworks like LangChain are abstracting away.

---

# Final Recommendation

Each approach has its place:

- **Use ReAct prompting** to learn how agent reasoning and tool orchestration work under the hood.
- **Use raw function calling** when you want maximum control, minimal dependencies, and a production-ready implementation tied to a specific model provider.
- **Use LangChain tool calling** when you're building larger applications that benefit from framework integrations, provider portability, and reduced boilerplate.

In practice, **native function calling is the sweet spot for many production applications**, offering a balance between explicit control and reliability. **LangChain becomes increasingly valuable as your application grows in complexity**, while **manual ReAct prompting remains the best educational tool for understanding the fundamentals of AI agents.**
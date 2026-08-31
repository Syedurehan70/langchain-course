# LangChain Search Engine Agent

A ReAct agent that searches the web via Tavily and returns structured results with sources.

Part of a LangChain course. Built with Groq instead of OpenAI.

## Stack

- **LangChain / LangGraph**: agent orchestration
- **Groq** (`openai/gpt-oss-20b`): LLM inference
- **Tavily**: web search tool
- **Pydantic**: response schema
- **LangSmith**: tracing

## How it works

1. The agent runs a ReAct loop, calling the `search` tool until it has enough information.
2. Raw results are pulled from the `ToolMessage` objects in the message chain.
3. A second, tool-free LLM call converts that text into JSON matching `AgentResponse`.
4. The JSON is parsed and validated into the Pydantic model.

The two-step split exists because Groq rejects structured output and tool calling in the same request.

## Setup

```bash
pip install langchain langgraph langchain-groq langchain-tavily python-dotenv pydantic
```

Create a `.env` file:

## Experiment: `main_toolstrategy.py`

An alternative implementation using LangChain's `ToolStrategy` for structured
output, instead of the two-step split in `main.py`.

**Why it's not the default:** it doesn't run reliably on Groq's free tier.

`ToolStrategy` is the correct approach in principle, it does structured output
through tool calling rather than JSON mode, which is what Groq supports
alongside tools. It also populates `structured_response`, so the schema shows
up properly in LangSmith. `main.py` parses JSON manually, so LangSmith only
logs a plain string.

In practice `openai/gpt-oss-20b` fails it, differently each run:

- **Malformed JSON**: produced a complete, correct `AgentResponse` with a stray
  `)` appended, which Groq rejected at the API boundary. LangChain's built-in
  `handle_errors` retry never fires, because the request is rejected before
  LangChain sees the response.
- **Non-termination**: with two tools available (`search` and `AgentResponse`),
  the model keeps searching instead of calling the schema tool, hitting
  `GraphRecursionError` even at `recursion_limit=20`.

A system prompt instructing it to search at most twice fixes the looping some of
the time, not consistently.

Worth revisiting on a stronger model, the design is sound, the model is the
constraint.
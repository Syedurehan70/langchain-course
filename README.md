# LangChain Search Engine Agent

A ReAct agent that searches the web via Tavily and returns structured results with sources.

Part of a LangChain course. Built with Groq instead of OpenAI.

## Stack

- **LangChain / LangGraph** — agent orchestration
- **Groq** (`openai/gpt-oss-20b`) — LLM inference
- **Tavily** — web search tool
- **Pydantic** — response schema
- **LangSmith** — tracing

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
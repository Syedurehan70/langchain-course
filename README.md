# LangChain Course

My work through a LangChain / LangGraph course, rebuilt to run on free
providers: **Groq** for hosted inference, **Ollama** for local models and
embeddings, and **Pinecone**'s free tier for vector storage. The instructor's
material assumes paid OpenAI keys; every project here documents what had to
change and why.

| # | Project | What it covers | Stack |
|---|---------|----------------|-------|
| 01 | [hello_langchain](01_hello_langchain) | First chain: `PromptTemplate \| LLM` | Groq |
| 02 | [search_engine](02_search_engine) | Tool-calling agent with web search and structured output | Groq, Tavily, LangSmith |
| 03 | [react_agents_in_detail](03_react_agents_in_detail) | One agent built three ways, peeling off abstraction layers | Ollama (qwen2.5:7b) |
| 04 | [intro_to_rag](04_intro_to_rag) | Ingestion and retrieval over a blog post | Ollama embeddings, Pinecone |

## Setup

The repo is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/):
one shared virtual environment and one lock file at the root, while each
project folder declares its own dependencies in its own `pyproject.toml`.

```powershell
uv python install 3.12
uv sync --all-packages
copy .env.example .env      # then fill in the keys
.venv\Scripts\python 01_hello_langchain\main.py
```

Add a dependency to one project:

```powershell
uv add --package search-engine <package>
```

## History

Each project was originally developed on its own branch. Those branches were
merged into `main` with their full commit history, so `git log` shows how each
one evolved; commits before the merge refer to files at their old, top-level
paths.

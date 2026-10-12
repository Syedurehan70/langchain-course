# llm_factory

Shared model configuration for every project in this repo.

```python
from dotenv import load_dotenv
load_dotenv()

from llm_factory import get_llm, get_embeddings, supports

llm = get_llm("agent")                 # reads LLM_AGENT, falls back to LLM_DEFAULT
embeddings = get_embeddings()          # reads EMBEDDINGS

if not supports("agent", "tools_with_structured_output"):
    ...                                # the calling code decides what to do
```

Roles are configured in `.env` as `provider:model`, for example
`LLM_AGENT=groq:openai/gpt-oss-20b` or `LLM_STRUCTURER=ollama:qwen2.5:7b`.
Switching provider is a one-line `.env` change; no code edits.

`KNOWN_LIMITS` lists provider limitations hit during the course. The factory
only reports them; it never silently changes strategy.

To use it from a project, add `"llm-factory"` to its dependencies and
`llm-factory = { workspace = true }` under `[tool.uv.sources]`.

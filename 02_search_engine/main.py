from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.errors import GraphRecursionError


class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


_tavily = TavilySearch(max_results=3)

MAX_CONTENT_CHARS = 400


@tool
def search(query: str) -> str:
    """Search the web. Takes plain text keywords, not a URL."""
    raw = _tavily.invoke({"query": query})

    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        return str(raw)[:1500]

    trimmed = [
        {
            "url": r.get("url", ""),
            "title": r.get("title", ""),
            "content": (r.get("content") or "")[:MAX_CONTENT_CHARS],
        }
        for r in raw.get("results", [])[:3]
    ]
    return json.dumps(trimmed)


llm = ChatGroq(model="openai/gpt-oss-20b")
agent = create_agent(model=llm, tools=[search])
plain_llm = ChatGroq(model="openai/gpt-oss-20b")

QUERY = (
    "Find 3 job postings for an AI engineer using LangChain in the Bay Area. "
    "Prefer LinkedIn results."
)


def run(query: str) -> AgentResponse:
    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"recursion_limit": 14},
    )

    tool_output = "\n\n".join(
        m.content for m in result["messages"] if isinstance(m, ToolMessage)
    )
    if not tool_output.strip():
        raise RuntimeError("No tool output — the agent never called search")

    prompt = f"""Below are raw web search results.

        Answer this request using only these results: {query}
        
        Return ONLY a JSON object in this exact shape:
        {{"answer": "<your answer>", "sources": [{{"url": "<source url>"}}]}}
        
        If the results don't satisfy the request, say so in the answer field.
        
        Search results:
        {tool_output}"""

    response = plain_llm.invoke([
        SystemMessage(content="You output only valid JSON. No markdown, no commentary."),
        HumanMessage(content=prompt),
    ])

    clean = (
        response.content.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    return AgentResponse(**json.loads(clean))


def main():
    result = run(QUERY)
    print(result.answer)
    print("\nSources:")
    for s in result.sources:
        print(" -", s.url)


if __name__ == "__main__":
    main()
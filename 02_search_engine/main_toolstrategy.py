from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json

load_dotenv()

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from groq import BadRequestError


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

agent = create_agent(
    model=llm,
    tools=[search],
    response_format=ToolStrategy(AgentResponse),
    system_prompt=(
        "Search at most twice, then immediately return your final answer "
        "using the AgentResponse format. Do not keep searching."
    ),
)

QUERY = (
    "Find 3 job postings for an AI engineer using LangChain in the Bay Area. "
    "Prefer LinkedIn results."
)


def main():
    last_error = None
    for attempt in range(3):
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=QUERY)]},
                config={"recursion_limit": 20},
            )
            break
        except BadRequestError as e:
            print(f"Attempt {attempt + 1} failed: malformed tool call, retrying")
            last_error = e
    else:
        raise last_error

    print("=== KEYS IN RESULT ===")
    print(list(result.keys()))

    structured = result.get("structured_response")
    if structured is None:
        print("No structured_response:")
        print(result["messages"][-1].content)
        return


    print("\n=== STRUCTURED TYPE ===")
    print(type(structured))

    if structured is None:
        print("\nNo structured_response — falling back to last message:")
        print(result["messages"][-1].content)
        return

    print("\n=== ANSWER ===")
    print(structured.answer)
    print("\n=== SOURCES ===")
    for s in structured.sources:
        print(" -", s.url)


if __name__ == "__main__":
    main()
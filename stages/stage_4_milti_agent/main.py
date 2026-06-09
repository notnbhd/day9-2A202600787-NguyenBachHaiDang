"""Stage 4: Multi-Agent System (In-Process)

Multiple specialised agents collaborate on a complex legal question
about Vietnamese drug law. Each specialist focuses on a specific domain
and uses Weaviate hybrid search for grounded retrieval.

Graph topology:
    analyze_law -> check_routing -> parallel [case_news, law_detail, drug_prevention]
        -> aggregate -> END

Improvements over Stage 3:
  - Specialist agents with domain-specific prompts
  - Parallel execution via LangGraph Send API
  - Conditional routing based on question analysis
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

import weaviate

from common.llm import get_llm

# ---------------------------------------------------------------------------
# Weaviate connection & embedding helpers (shared by all agents)
# ---------------------------------------------------------------------------

WEAVIATE_COLLECTION = os.getenv("WEAVIATE_COLLECTION", "DrugLawDocs")
HYBRID_ALPHA = 0.75


def _get_weaviate_client() -> weaviate.WeaviateClient:
    """Connect to Weaviate (local or cloud)."""
    weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY", "")

    if weaviate_api_key:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key),
        )
    else:
        client = weaviate.connect_to_local(
            host=weaviate_url.replace("http://", "").replace("https://", "").split(":")[0],
            port=int(weaviate_url.split(":")[-1]) if ":" in weaviate_url.rsplit("/", 1)[-1] else 8080,
        )
    return client


def _embed_query(text: str) -> list[float]:
    """Compute a 1536-dim embedding via OpenRouter (text-embedding-3-small)."""
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
    )
    return embeddings.embed_query(text)


def _weaviate_hybrid(query: str, limit: int = 3, doc_type_filter: str | None = None) -> str:
    """Shared Weaviate hybrid search helper used by specialist tools."""
    try:
        from weaviate.classes.query import Filter, MetadataQuery

        query_vector = _embed_query(query)
        client = _get_weaviate_client()
        collection = client.collections.get(WEAVIATE_COLLECTION)

        kwargs = {
            "query": query,
            "vector": query_vector,
            "alpha": HYBRID_ALPHA,
            "limit": limit,
            "return_metadata": MetadataQuery(score=True),
        }
        if doc_type_filter:
            kwargs["filters"] = Filter.by_property("doc_type").equal(doc_type_filter)

        results = collection.query.hybrid(**kwargs)

        if results.objects:
            formatted = []
            for i, obj in enumerate(results.objects, 1):
                props = obj.properties
                source = props.get("source", "unknown")
                chunk_idx = props.get("chunk_index", "?")
                content = props.get("content", "")
                score = obj.metadata.score
                score_str = f", score={score:.4f}" if score is not None else ""
                formatted.append(
                    f"[Result {i} | {source} (chunk {chunk_idx}{score_str})]\n{content}"
                )
            client.close()
            return "\n\n---\n\n".join(formatted)

        client.close()
        return "Không tìm thấy kết quả phù hợp."
    except Exception as e:
        return f"Lỗi truy vấn Weaviate: {e}"


# ---------------------------------------------------------------------------
# Tools for specialist sub-agents
# ---------------------------------------------------------------------------

@tool
def search_criminal_law(query: str) -> str:
    """Tra cứu Bộ luật Hình sự Việt Nam — các tội danh, khung hình phạt,
    tình tiết tăng nặng/giảm nhẹ liên quan đến ma túy.

    Args:
        query: Câu hỏi về luật hình sự (tội danh, mức án, điều luật).
    """
    return _weaviate_hybrid(query, limit=3, doc_type_filter="legal")


@tool
def search_drug_news(query: str) -> str:
    """Tra cứu các vụ án ma túy thực tế tại Việt Nam từ báo chí —
    các vụ nghệ sĩ, người nổi tiếng bị bắt, mức án, chi tiết vụ việc.

    Args:
        query: Từ khóa tìm kiếm (tên người, loại tội, vụ việc cụ thể).
    """
    return _weaviate_hybrid(query, limit=5, doc_type_filter="news")


@tool
def search_drug_prevention_law(query: str) -> str:
    """Tra cứu Luật Phòng chống ma túy Việt Nam (2021, 2025) — các quy định
    về phòng ngừa, cai nghiện, quản lý người sử dụng ma túy, hiệu lực.

    Args:
        query: Câu hỏi về luật phòng chống ma túy.
    """
    enriched = f"{query} phòng chống ma túy luật"
    return _weaviate_hybrid(enriched, limit=3, doc_type_filter="legal")


@tool
def check_law_effective_date(law_name: str) -> str:
    """Tra cứu thời gian bắt đầu có hiệu lực thi hành của bộ luật.

    Args:
        law_name: Tên bộ luật (ví dụ: 'Bộ luật Hình sự', 'Luật PCMT 2025').
    """
    enriched = f"{law_name} hiệu lực thi hành ngày có hiệu lực"
    return _weaviate_hybrid(enriched, limit=3, doc_type_filter="legal")


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

from typing import Annotated, TypedDict

from langgraph.constants import Send
from langgraph.graph import END, StateGraph


def _last_wins(a: str, b: str) -> str:
    """Reducer: keep the most recently written value."""
    return b if b else a


class LegalState(TypedDict):
    question: str
    law_analysis: str
    needs_case_news: bool
    needs_law_detail: bool
    needs_drug_prevention: bool
    case_news_result: Annotated[str, _last_wins]
    law_detail_result: Annotated[str, _last_wins]
    drug_prevention_result: Annotated[str, _last_wins]
    final_answer: str


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

async def analyze_law(state: LegalState) -> dict:
    """Lead attorney analyses the legal aspects of the question."""
    print("\n  [Node: analyze_law] Chuyên gia pháp luật phân tích câu hỏi...")
    llm = get_llm()
    messages = [
        SystemMessage(
            content=(
                "Bạn là luật sư hình sự cấp cao chuyên về pháp luật Việt Nam, "
                "đặc biệt các tội liên quan đến ma túy. Phân tích câu hỏi pháp lý "
                "dưới đây, xác định các khía cạnh cần tra cứu sâu hơn. "
                "Giữ phân tích dưới 200 từ."
            )
        ),
        HumanMessage(content=state["question"]),
    ]
    result = await llm.ainvoke(messages)
    print(f"  [Node: analyze_law] Hoàn thành ({len(result.content)} ký tự)")
    return {"law_analysis": result.content}


async def check_routing(state: LegalState) -> dict:
    """Routing node: determine which specialist sub-agents are needed."""
    print("\n  [Node: check_routing] Xác định cần gọi chuyên gia nào...")
    llm = get_llm()
    messages = [
        SystemMessage(
            content=(
                'Bạn là chuyên gia routing. Dựa vào câu hỏi và phân tích ban đầu, '
                'quyết định cần gọi chuyên gia nào.\n'
                'Trả lời CHỈDUY NHẤT JSON hợp lệ — không markdown, không text thừa:\n'
                '{"needs_case_news": <true|false>, "needs_law_detail": <true|false>, '
                '"needs_drug_prevention": <true|false>}\n\n'
                'needs_case_news = true → câu hỏi liên quan đến vụ án thực tế, nghệ sĩ, '
                'người nổi tiếng bị bắt vì ma túy\n'
                'needs_law_detail = true → cần tra cứu điều luật cụ thể, khung hình phạt, '
                'tội danh trong Bộ luật Hình sự\n'
                'needs_drug_prevention = true → cần thông tin về Luật Phòng chống ma túy, '
                'cai nghiện, hiệu lực luật mới'
            )
        ),
        HumanMessage(
            content=f"Câu hỏi: {state['question']}\n\n"
                    f"Phân tích sơ bộ: {state.get('law_analysis', 'N/A')}"
        ),
    ]
    result = await llm.ainvoke(messages)
    raw = result.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Default: all specialists
        parsed = {"needs_case_news": True, "needs_law_detail": True, "needs_drug_prevention": True}

    needs_case_news = bool(parsed.get("needs_case_news", False))
    needs_law_detail = bool(parsed.get("needs_law_detail", True))
    needs_drug_prevention = bool(parsed.get("needs_drug_prevention", False))

    print(f"  [Node: check_routing] case_news={needs_case_news}, "
          f"law_detail={needs_law_detail}, drug_prevention={needs_drug_prevention}")
    return {
        "needs_case_news": needs_case_news,
        "needs_law_detail": needs_law_detail,
        "needs_drug_prevention": needs_drug_prevention,
    }


def route_to_specialists(state: LegalState) -> list[Send]:
    """Routing function: dispatch parallel Send objects to specialist nodes."""
    sends: list[Send] = []
    if state.get("needs_case_news"):
        sends.append(Send("call_case_news_specialist", state))
    if state.get("needs_law_detail"):
        sends.append(Send("call_law_detail_specialist", state))
    if state.get("needs_drug_prevention"):
        sends.append(Send("call_drug_prevention_specialist", state))
    if not sends:
        sends.append(Send("aggregate", state))
    return sends


async def call_case_news_specialist(state: LegalState) -> dict:
    """Specialist: tra cứu vụ án thực tế từ báo chí (ReAct agent)."""
    from langgraph.prebuilt import create_react_agent

    print("\n  [Node: call_case_news_specialist] Chuyên gia vụ án thực tế đang phân tích...")

    prompt = (
        "Bạn là nhà báo pháp luật chuyên theo dõi các vụ án ma túy tại Việt Nam. "
        "Sử dụng tool search_drug_news để tìm các vụ án thực tế liên quan. "
        "Trích dẫn tên người, mức án, thời gian. Trả lời dưới 200 từ bằng tiếng Việt."
    )

    llm = get_llm()
    agent = create_react_agent(model=llm, tools=[search_drug_news], prompt=prompt)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": state["question"]}]})

    final_msg = result["messages"][-1].content
    print(f"  [Node: call_case_news_specialist] Hoàn thành ({len(final_msg)} ký tự)")
    return {"case_news_result": final_msg}


async def call_law_detail_specialist(state: LegalState) -> dict:
    """Specialist: tra cứu điều luật hình sự cụ thể (ReAct agent)."""
    from langgraph.prebuilt import create_react_agent

    print("\n  [Node: call_law_detail_specialist] Chuyên gia luật hình sự đang phân tích...")

    prompt = (
        "Bạn là luật sư hình sự cấp cao tại Việt Nam, chuyên về các tội liên quan đến "
        "ma túy trong Bộ luật Hình sự. Sử dụng tool search_criminal_law và "
        "check_law_effective_date để tra cứu điều luật cụ thể, khung hình phạt, "
        "và hiệu lực thi hành. Trích dẫn số điều, khoản cụ thể. "
        "Trả lời dưới 200 từ bằng tiếng Việt."
    )

    llm = get_llm()
    agent = create_react_agent(
        model=llm, tools=[search_criminal_law, check_law_effective_date], prompt=prompt
    )
    result = await agent.ainvoke({"messages": [{"role": "user", "content": state["question"]}]})

    final_msg = result["messages"][-1].content
    print(f"  [Node: call_law_detail_specialist] Hoàn thành ({len(final_msg)} ký tự)")
    return {"law_detail_result": final_msg}


async def call_drug_prevention_specialist(state: LegalState) -> dict:
    """Specialist: tra cứu Luật Phòng chống ma túy (ReAct agent)."""
    from langgraph.prebuilt import create_react_agent

    print("\n  [Node: call_drug_prevention_specialist] Chuyên gia Luật PCMT đang phân tích...")

    prompt = (
        "Bạn là chuyên gia về Luật Phòng chống ma túy Việt Nam (2021, 2025). "
        "Sử dụng tool search_drug_prevention_law và check_law_effective_date để "
        "tra cứu các quy định phòng ngừa, cai nghiện, quản lý người sử dụng ma túy, "
        "và hiệu lực của các luật liên quan. Trả lời dưới 200 từ bằng tiếng Việt."
    )

    llm = get_llm()
    agent = create_react_agent(
        model=llm, tools=[search_drug_prevention_law, check_law_effective_date], prompt=prompt
    )
    result = await agent.ainvoke({"messages": [{"role": "user", "content": state["question"]}]})

    final_msg = result["messages"][-1].content
    print(f"  [Node: call_drug_prevention_specialist] Hoàn thành ({len(final_msg)} ký tự)")
    return {"drug_prevention_result": final_msg}


async def aggregate(state: LegalState) -> dict:
    """Combine all specialist analyses into a final comprehensive answer."""
    print("\n  [Node: aggregate] Tổng hợp kết quả từ các chuyên gia...")
    llm = get_llm()

    sections: list[str] = []
    if state.get("law_analysis"):
        sections.append(f"## Phân tích pháp lý tổng quan\n{state['law_analysis']}")
    if state.get("law_detail_result"):
        sections.append(f"## Chi tiết luật Hình sự\n{state['law_detail_result']}")
    if state.get("case_news_result"):
        sections.append(f"## Vụ án thực tế\n{state['case_news_result']}")
    if state.get("drug_prevention_result"):
        sections.append(f"## Luật Phòng chống ma túy\n{state['drug_prevention_result']}")

    combined = "\n\n---\n\n".join(sections)

    messages = [
        SystemMessage(
            content=(
                "Bạn là cố vấn pháp luật cấp cao, tổng hợp các phân tích chuyên sâu "
                "từ nhiều chuyên gia thành một câu trả lời toàn diện, dễ hiểu. "
                "Kết hợp thông tin từ các nguồn, tránh lặp lại. Trích dẫn điều luật cụ thể. "
                "Trả lời bằng tiếng Việt, dưới 500 từ."
            )
        ),
        HumanMessage(content=combined),
    ]
    result = await llm.ainvoke(messages)
    print(f"  [Node: aggregate] Hoàn thành ({len(result.content)} ký tự)")
    return {"final_answer": result.content}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def create_graph():
    """Build and compile the multi-agent StateGraph."""
    graph = StateGraph(LegalState)

    graph.add_node("analyze_law", analyze_law)
    graph.add_node("check_routing", check_routing)
    graph.add_node("call_case_news_specialist", call_case_news_specialist)
    graph.add_node("call_law_detail_specialist", call_law_detail_specialist)
    graph.add_node("call_drug_prevention_specialist", call_drug_prevention_specialist)
    graph.add_node("aggregate", aggregate)

    graph.set_entry_point("analyze_law")
    graph.add_edge("analyze_law", "check_routing")
    graph.add_conditional_edges(
        "check_routing",
        route_to_specialists,
        ["call_case_news_specialist", "call_law_detail_specialist",
         "call_drug_prevention_specialist", "aggregate"],
    )
    graph.add_edge("call_case_news_specialist", "aggregate")
    graph.add_edge("call_law_detail_specialist", "aggregate")
    graph.add_edge("call_drug_prevention_specialist", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()


QUESTION = (
    "Ca sĩ Châu Việt Cường phạm tội gì và bị kết án bao nhiêu năm? "
    "Tội danh thuộc điều nào trong Bộ luật Hình sự? "
    "Luật Phòng chống ma túy mới nhất 2025 có gì thay đổi so với 2021?"
)


async def main():
    print("=" * 70)
    print("STAGE 4: Multi-Agent System (In-Process)")
    print("=" * 70)
    print()
    print("[How it works]")
    print("  1. Lead attorney agent analyses the question")
    print("  2. Router decides which specialist agents are needed")
    print("  3. Specialists run IN PARALLEL (LangGraph Send API):")
    print("     - Case News Specialist: tra cứu vụ án thực tế từ báo chí")
    print("     - Law Detail Specialist: tra cứu Bộ luật Hình sự")
    print("     - Drug Prevention Specialist: tra cứu Luật Phòng chống ma túy")
    print("  4. Aggregator combines all analyses into a final answer")
    print()
    print("[Graph topology]")
    print("  analyze_law -> check_routing -> [case_news + law_detail + drug_prevention]")
    print("                                          -> aggregate -> END")
    print()
    print(f"Question: {QUESTION}")
    print("-" * 70)

    graph = create_graph()

    result = await graph.ainvoke({
        "question": QUESTION,
        "law_analysis": "",
        "needs_case_news": False,
        "needs_law_detail": False,
        "needs_drug_prevention": False,
        "case_news_result": "",
        "law_detail_result": "",
        "drug_prevention_result": "",
        "final_answer": "",
    })

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(result["final_answer"])

    print()
    print("-" * 70)
    print("[Improvements over Stage 3]")
    print("  + Specialisation: each agent has domain-specific expertise")
    print("  + Parallel execution: specialists run concurrently via Send API")
    print("  + Better quality: specialist prompts produce deeper analysis")
    print("  + Weaviate RAG: all tools grounded in real Vietnamese law database")
    print("  + Conditional routing: only relevant specialists are dispatched")
    print()
    print("[Stage 4 (Monolith) vs Stage 5 (Distributed A2A)]")
    print("  +---------------------------+-------------------------------+")
    print("  | Stage 4 (In-Process)      | Stage 5 (A2A Protocol)        |")
    print("  +---------------------------+-------------------------------+")
    print("  | Single process            | Multiple services (ports)     |")
    print("  | Direct function calls     | HTTP-based A2A protocol       |")
    print("  | Shared memory             | Message passing               |")
    print("  | Simple deployment         | Independent scaling           |")
    print("  | Tight coupling            | Loose coupling                |")
    print("  | Easy to debug             | Service discovery + registry  |")
    print("  | Good for small teams      | Good for large organisations  |")
    print("  +---------------------------+-------------------------------+")
    print()
    print("Stage 5 (this repo's main project) takes this same graph topology")
    print("and deploys each agent as an independent A2A service. Run it with:")
    print("  ./start_all.sh && python test_client.py")
    print("=" * 70)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
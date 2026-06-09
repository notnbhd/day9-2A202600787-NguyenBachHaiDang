"""Stage 3: Single Agent (ReAct Loop)

Wraps the LLM + tools in an autonomous agent that can reason, act,
and observe in a loop. The agent decides which tools to call, evaluates
the results, and may call more tools before giving a final answer.

Uses LangGraph's create_react_agent for the Think -> Act -> Observe loop.

Improvements over Stage 2:
  - Tools now query Weaviate 'DrugLawDocs' collection via hybrid search
  - Agent autonomously decides tool ordering and iteration
  - Multi-step reasoning: search law → search news → check dates → answer
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
from langchain_core.tools import tool

import weaviate

from common.llm import get_llm

# ---------------------------------------------------------------------------
# Weaviate vector-store connection
# ---------------------------------------------------------------------------

WEAVIATE_COLLECTION = os.getenv("WEAVIATE_COLLECTION", "DrugLawDocs")
HYBRID_ALPHA = 0.75  # 0 = pure BM25, 1 = pure vector


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


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def search_legal_database(query: str) -> str:
    """Tra cứu cơ sở dữ liệu pháp luật Việt Nam để tìm điều luật, quy định
    hình sự liên quan đến ma túy. Sử dụng Weaviate hybrid search (BM25 + vector).

    Args:
        query: Câu hỏi hoặc từ khóa pháp lý cần tra cứu.
    """
    try:
        from weaviate.classes.query import Filter, MetadataQuery

        query_vector = _embed_query(query)
        client = _get_weaviate_client()
        collection = client.collections.get(WEAVIATE_COLLECTION)

        results = collection.query.hybrid(
            query=query,
            vector=query_vector,
            alpha=HYBRID_ALPHA,
            limit=3,
            filters=Filter.by_property("doc_type").equal("legal"),
            return_metadata=MetadataQuery(score=True),
        )

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
    except Exception as e:
        print(f"  ⚠ Weaviate unavailable ({e})")

    return "Không tìm thấy văn bản pháp luật liên quan."


@tool
def search_drug_case_news(query: str) -> str:
    """Tra cứu các vụ án ma túy thực tế tại Việt Nam từ cơ sở dữ liệu
    báo chí. Bao gồm các vụ nghệ sĩ, người nổi tiếng bị bắt hoặc bị
    khởi tố liên quan đến ma túy — kết quả xét xử, mức án, chi tiết vụ việc.

    Args:
        query: Từ khóa tìm kiếm (tên người, loại tội, hoặc mô tả vụ việc).
    """
    try:
        from weaviate.classes.query import Filter, MetadataQuery

        query_vector = _embed_query(query)
        client = _get_weaviate_client()
        collection = client.collections.get(WEAVIATE_COLLECTION)

        results = collection.query.hybrid(
            query=query,
            vector=query_vector,
            alpha=HYBRID_ALPHA,
            limit=5,
            filters=Filter.by_property("doc_type").equal("news"),
            return_metadata=MetadataQuery(score=True),
        )

        if results.objects:
            formatted = []
            for i, obj in enumerate(results.objects, 1):
                props = obj.properties
                source = props.get("source", "unknown")
                content = props.get("content", "")
                score = obj.metadata.score
                score_str = f", score={score:.4f}" if score is not None else ""
                formatted.append(
                    f"[Vụ án {i} | {source}{score_str}]\n{content}"
                )
            client.close()
            return "\n\n---\n\n".join(formatted)

        client.close()
        return "Không tìm thấy vụ án liên quan trong cơ sở dữ liệu báo chí."
    except Exception as e:
        print(f"  ⚠ Weaviate unavailable ({e})")
        return "Không thể kết nối cơ sở dữ liệu báo chí lúc này."


@tool
def check_law_effective_date(law_name: str) -> str:
    """Tra cứu thời gian bắt đầu có hiệu lực thi hành của các bộ luật
    liên quan đến ma túy và hình sự tại Việt Nam.

    Args:
        law_name: Tên hoặc mô tả bộ luật cần tra cứu (ví dụ: 'Bộ luật Hình sự',
                  'Luật Phòng chống ma túy 2021', 'Luật 73/2021/QH14').
    """
    try:
        from weaviate.classes.query import Filter, MetadataQuery

        enriched_query = f"{law_name} hiệu lực thi hành ngày có hiệu lực"
        query_vector = _embed_query(enriched_query)
        client = _get_weaviate_client()
        collection = client.collections.get(WEAVIATE_COLLECTION)

        results = collection.query.hybrid(
            query=enriched_query,
            vector=query_vector,
            alpha=HYBRID_ALPHA,
            limit=5,
            filters=Filter.by_property("doc_type").equal("legal"),
            return_metadata=MetadataQuery(score=True),
        )

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
                    f"[Nguồn {i} | {source} (chunk {chunk_idx}{score_str})]\n{content}"
                )
            client.close()
            return "\n\n---\n\n".join(formatted)

        client.close()
        return "Không tìm thấy thông tin hiệu lực của bộ luật này."
    except Exception as e:
        print(f"  ⚠ Weaviate unavailable ({e})")
        return "Không thể kết nối cơ sở dữ liệu pháp luật lúc này."


@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ án.

    Args:
        case_type: Loại vụ án (contract, tort, property, drug, labor)
    """
    limits = {
        "contract": "4 năm (UCC § 2-725)",
        "tort": "2-3 năm tùy bang",
        "property": "5 năm",
        "drug": "Theo Bộ luật Hình sự VN 2015: 5-20 năm tùy khung hình phạt (Điều 27 BLHS)",
        "labor": "1 năm kể từ ngày phát hiện vi phạm (Điều 190 BLLĐ 2019)",
    }
    return limits.get(case_type.lower(), "Không xác định — vui lòng chỉ rõ loại vụ án.")


# --- Bài Tập 3.1: Tool tra cứu án lệ từ Weaviate ---
@tool
def search_case_law(keywords: str) -> str:
    """Tìm kiếm án lệ và tiền lệ pháp lý theo từ khóa. Tra cứu trong
    toàn bộ cơ sở dữ liệu (cả văn bản pháp luật lẫn báo chí) để tìm
    các vụ án cụ thể, phán quyết, hoặc mức hình phạt đã áp dụng.

    Args:
        keywords: Từ khóa tìm kiếm (tên tội danh, tên người, mức án, v.v.).
    """
    try:
        from weaviate.classes.query import MetadataQuery

        query_vector = _embed_query(keywords)
        client = _get_weaviate_client()
        collection = client.collections.get(WEAVIATE_COLLECTION)

        # Search across ALL doc types (legal + news) for case precedents
        results = collection.query.hybrid(
            query=keywords,
            vector=query_vector,
            alpha=HYBRID_ALPHA,
            limit=4,
            return_metadata=MetadataQuery(score=True),
        )

        if results.objects:
            formatted = []
            for i, obj in enumerate(results.objects, 1):
                props = obj.properties
                source = props.get("source", "unknown")
                doc_type = props.get("doc_type", "unknown")
                content = props.get("content", "")
                score = obj.metadata.score
                score_str = f", score={score:.4f}" if score is not None else ""
                tag = "Văn bản PL" if doc_type == "legal" else "Báo chí"
                formatted.append(
                    f"[Án lệ {i} | {tag} | {source}{score_str}]\n{content}"
                )
            client.close()
            return "\n\n---\n\n".join(formatted)

        client.close()
        return "Không tìm thấy án lệ phù hợp."
    except Exception as e:
        print(f"  ⚠ Weaviate unavailable ({e})")
        return "Không thể kết nối cơ sở dữ liệu lúc này."


TOOLS = [
    search_legal_database,
    search_drug_case_news,
    check_law_effective_date,
    check_statute_of_limitations,
    search_case_law,
]

QUESTION = (
    "Châu Việt Cường bị kết án bao nhiêu năm tù? "
    "Tội danh của anh ấy thuộc điều nào trong Bộ luật Hình sự? "
    "Luật Phòng chống ma túy mới nhất có hiệu lực từ ngày nào?"
)

SYSTEM_PROMPT = (
    "Bạn là chuyên gia pháp luật Việt Nam, đặc biệt về luật hình sự liên quan "
    "đến ma túy. Bạn có các công cụ:\n"
    "  - search_legal_database: tra cứu điều luật (Bộ luật Hình sự, Luật PCMT)\n"
    "  - search_drug_case_news: tra cứu vụ án thực tế từ báo chí\n"
    "  - check_law_effective_date: kiểm tra ngày hiệu lực của bộ luật\n"
    "  - check_statute_of_limitations: kiểm tra thời hiệu khởi kiện\n"
    "  - search_case_law: tìm kiếm án lệ / tiền lệ pháp lý\n\n"
    "Hãy sử dụng các tool phù hợp để xây dựng câu trả lời có căn cứ. "
    "Luôn trích dẫn điều luật cụ thể. Trả lời bằng tiếng Việt, dưới 500 từ."
)


async def main():
    from langgraph.prebuilt import create_react_agent

    print("=" * 70)
    print("STAGE 3: Single Agent (ReAct Loop)")
    print("=" * 70)
    print()
    print("[How it works]")
    print("  1. An autonomous agent receives a complex multi-part question")
    print("  2. It reasons about what tools to call (Think)")
    print("  3. It calls a tool (Act)")
    print("  4. It observes the result and decides next steps (Observe)")
    print("  5. It repeats until it has enough information for a final answer")
    print()
    print(f"Question: {QUESTION}")
    print("-" * 70)

    llm = get_llm()
    graph = create_react_agent(model=llm, tools=TOOLS, prompt=SYSTEM_PROMPT)

    inputs = {"messages": [{"role": "user", "content": QUESTION}]}

    step = 0
    async for chunk in graph.astream(inputs, stream_mode="updates"):
        for node_name, update in chunk.items():
            step += 1
            messages = update.get("messages", [])
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    print(f"\n[Step {step}] THINK + ACT (node: {node_name})")
                    for tc in msg.tool_calls:
                        print(f"  Tool: {tc['name']}")
                        print(f"  Args: {tc['args']}")
                elif msg.type == "tool":
                    print(f"\n[Step {step}] OBSERVE (node: {node_name})")
                    content = msg.content
                    print(f"  Result: {content[:300]}{'...' if len(content) > 300 else ''}")
                elif msg.type == "ai" and msg.content:
                    print(f"\n[Step {step}] FINAL ANSWER (node: {node_name})")
                    print("-" * 70)
                    print(msg.content)

    print()
    print("-" * 70)
    print("[Improvements over Stage 2]")
    print("  + Autonomous: agent decides which tools to call and when")
    print("  + Multi-step reasoning: can search, calculate, search again")
    print("  + Handles complex queries: breaks problems into sub-tasks")
    print("  + Weaviate RAG: all tools grounded in real Vietnamese law database")
    print()
    print("[Limitations of Stage 3]")
    print("  - Single agent: one LLM handles all domains")
    print("  - No specialisation: same system prompt for all legal areas")
    print("  - Bottleneck: sequential tool calls, no parallelism")
    print()
    print("Next: Stage 4 splits this into specialised agents that work in parallel.")
    print("=" * 70)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
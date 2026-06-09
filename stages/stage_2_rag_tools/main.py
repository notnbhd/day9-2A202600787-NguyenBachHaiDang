"""Stage 2: LLM + RAG / Tools

Adds retrieval-augmented generation and tool use to ground LLM responses
in external data. The LLM can now search a Weaviate vector database of
Vietnamese drug-law documents and calculate damages — but the orchestration
is manual (one tool-call loop).
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from common.llm import get_llm

# ---------------------------------------------------------------------------
# Weaviate vector-store connection
# ---------------------------------------------------------------------------

import weaviate


def _get_weaviate_client() -> weaviate.WeaviateClient:
    """Connect to Weaviate (local or cloud)."""
    weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY", "")

    if weaviate_api_key:
        # Weaviate Cloud (WCD) or authenticated instance
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key),
        )
    else:
        # Local Weaviate (docker, embedded, etc.)
        client = weaviate.connect_to_local(
            host=weaviate_url.replace("http://", "").replace("https://", "").split(":")[0],
            port=int(weaviate_url.split(":")[-1]) if ":" in weaviate_url.rsplit("/", 1)[-1] else 8080,
        )
    return client


# ---------------------------------------------------------------------------
# Fallback: simulated knowledge base (used when Weaviate is unavailable)
# ---------------------------------------------------------------------------

LEGAL_KNOWLEDGE = [
    {
        "id": "ucc_breach",
        "keywords": ["breach", "contract", "remedies", "damages", "ucc"],
        "text": (
            "Under the Uniform Commercial Code (UCC) Article 2, remedies for breach of contract "
            "include: (1) expectation damages — placing the non-breaching party in the position "
            "they would have been in had the contract been performed; (2) consequential damages "
            "for foreseeable losses (Hadley v. Baxendale, 1854); (3) specific performance when "
            "the subject matter is unique; (4) cover damages — the cost of obtaining substitute "
            "performance. The statute of limitations is typically 4 years (UCC § 2-725)."
        ),
    },
    {
        "id": "nda_trade_secret",
        "keywords": ["nda", "non-disclosure", "confidential", "trade secret", "agreement"],
        "text": (
            "NDA breaches may trigger both contractual and statutory liability. Under the Defend "
            "Trade Secrets Act (DTSA, 18 U.S.C. § 1836), misappropriation of trade secrets can "
            "result in: (1) injunctive relief; (2) actual damages plus unjust enrichment; "
            "(3) exemplary damages up to 2x actual damages for willful misappropriation; "
            "(4) attorney's fees. State Uniform Trade Secrets Act (UTSA) versions provide "
            "additional remedies. Criminal prosecution is possible under the Economic Espionage "
            "Act (18 U.S.C. § 1832) with penalties up to $5M for individuals."
        ),
    },
    {
        "id": "dtsa_details",
        "keywords": ["dtsa", "federal", "trade secret", "defend", "statute"],
        "text": (
            "The Defend Trade Secrets Act (2016) created a federal private cause of action for "
            "trade secret misappropriation. Key provisions: (1) ex parte seizure orders in "
            "extraordinary circumstances; (2) 3-year statute of limitations; (3) immunity for "
            "whistleblower disclosures to government officials; (4) employers must notify "
            "employees of whistleblower immunity in any NDA or employment agreement."
        ),
    },
    {
        "id": "liquidated_damages",
        "keywords": ["liquidated", "damages", "penalty", "clause", "contract", "nda"],
        "text": (
            "Liquidated damages clauses in NDAs are enforceable if: (1) actual damages would be "
            "difficult to calculate at the time of contracting; (2) the stipulated amount is a "
            "reasonable estimate of anticipated harm. Courts will void clauses that function as "
            "penalties (Restatement (Second) of Contracts § 356). Typical NDA liquidated damages "
            "range from $10,000 to $500,000 depending on the nature of the confidential information."
        ),
    },
    {
        "id": "injunctive_relief",
        "keywords": ["injunction", "restraining", "order", "equitable", "nda", "breach"],
        "text": (
            "Courts routinely grant temporary restraining orders (TROs) and preliminary injunctions "
            "for NDA breaches because: (1) confidential information, once disclosed, cannot be "
            "'un-disclosed' — making monetary damages inadequate; (2) irreparable harm is presumed "
            "for trade secret misappropriation in many jurisdictions. The movant must show "
            "likelihood of success on the merits, irreparable harm, balance of equities, and "
            "public interest (Winter v. Natural Resources Defense Council, 2008)."
        ),
    },
    # --- Bài Tập 2.1: Thêm knowledge base entry về luật lao động ---
    {
        "id": "labor_law",
        "keywords": ["lao động", "sa thải", "hợp đồng lao động", "labor", "termination"],
        "text": (
            "Theo Bộ luật Lao động Việt Nam 2019, người sử dụng lao động có thể "
            "đơn phương chấm dứt hợp đồng trong các trường hợp: (1) người lao động "
            "thường xuyên không hoàn thành công việc; (2) bị ốm đau, tai nạn đã điều trị "
            "12 tháng chưa khỏi; (3) thiên tai, hỏa hoạn; (4) người lao động đủ tuổi nghỉ hưu."
        ),
    },
]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

WEAVIATE_COLLECTION = os.getenv("WEAVIATE_COLLECTION", "DrugLawDocs")

# Hybrid search tuning
HYBRID_ALPHA = 0.5  # 0 = pure BM25, 1 = pure vector; 0.75 favours semantic


def _embed_query(text: str) -> list[float]:
    """Compute a 1536-dim embedding via OpenRouter (text-embedding-3-small)."""
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
    )
    return embeddings.embed_query(text)


@tool
def search_legal_database(query: str) -> str:
    """Search the legal knowledge base for relevant Vietnamese drug-law
    statutes, case law, and legal principles. Uses Weaviate hybrid search
    (BM25 + dense vector) when available, otherwise falls back to a local
    keyword-based knowledge base.

    Args:
        query: The legal question or search terms to look up.
    """
    # --- Try Weaviate hybrid search first ---
    try:
        from weaviate.classes.query import MetadataQuery

        query_vector = _embed_query(query)
        client = _get_weaviate_client()
        collection = client.collections.get(WEAVIATE_COLLECTION)

        # Hybrid search (BM25 + dense) — top 3 most relevant chunks
        results = collection.query.hybrid(
            query=query,
            vector=query_vector,
            alpha=HYBRID_ALPHA,
            limit=3,
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
        print(f"  ⚠ Weaviate unavailable ({e}), falling back to local knowledge base.")

    # --- Fallback: keyword search on local LEGAL_KNOWLEDGE ---
    query_words = set(query.lower().split())
    scored = []
    for entry in LEGAL_KNOWLEDGE:
        overlap = len(query_words & set(entry["keywords"]))
        if overlap > 0:
            scored.append((overlap, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:2]
    if not top:
        return "No relevant legal sources found for this query."
    results = []
    for _, entry in top:
        results.append(f"[{entry['id']}] {entry['text']}")
    return "\n\n".join(results)


@tool
def calculate_damages(breach_type: str, contract_value: float) -> str:
    """Calculate estimated damages for a contract breach based on type and contract value."""
    breach_type_lower = breach_type.lower()
    if "willful" in breach_type_lower or "intentional" in breach_type_lower:
        multiplier = 2.0
        label = "Willful/intentional breach (2x multiplier under DTSA)"
    elif "negligent" in breach_type_lower:
        multiplier = 1.0
        label = "Negligent breach (1x actual damages)"
    else:
        multiplier = 1.5
        label = "Standard breach (1.5x estimated multiplier)"

    base_damages = contract_value * multiplier
    attorney_fees = contract_value * 0.15
    total = base_damages + attorney_fees

    return (
        f"Damage Estimate:\n"
        f"  Breach type: {label}\n"
        f"  Contract value: ${contract_value:,.2f}\n"
        f"  Estimated damages: ${base_damages:,.2f}\n"
        f"  Attorney's fees (~15%): ${attorney_fees:,.2f}\n"
        f"  Total estimated exposure: ${total:,.2f}"
    )


# --- Bài Tập 2.2: Tool kiểm tra thời hiệu khởi kiện ---
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


# --- Tool tra cứu vụ án ma túy thực tế từ báo chí ---
@tool
def search_drug_case_news(query: str) -> str:
    """Tra cứu các vụ án ma túy thực tế tại Việt Nam từ cơ sở dữ liệu
    báo chí. Bao gồm các vụ nghệ sĩ, người nổi tiếng bị bắt hoặc bị
    khởi tố liên quan đến ma túy — kết quả xét xử, mức án, và chi tiết
    vụ việc.

    Sử dụng tool này khi cần tìm ví dụ thực tế, tiền lệ, hoặc thông tin
    về các vụ án ma túy cụ thể đã xảy ra.

    Args:
        query: Từ khóa tìm kiếm (tên người, loại tội, hoặc mô tả vụ việc).
    """
    try:
        from weaviate.classes.query import Filter, MetadataQuery

        query_vector = _embed_query(query)
        client = _get_weaviate_client()
        collection = client.collections.get(WEAVIATE_COLLECTION)

        # Hybrid search filtered to news articles only
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
        print(f"  ⚠ Weaviate unavailable ({e}), không thể tra cứu báo chí.")
        return "Không thể kết nối cơ sở dữ liệu báo chí lúc này."


# --- Tool kiểm tra hiệu lực thi hành của bộ luật ---
@tool
def check_law_effective_date(law_name: str) -> str:
    """Tra cứu thời gian bắt đầu có hiệu lực thi hành của các bộ luật
    liên quan đến ma túy và hình sự tại Việt Nam. Tìm kiếm trong cơ sở
    dữ liệu văn bản pháp luật để xác định ngày hiệu lực, các lần sửa đổi
    bổ sung, và tình trạng hiệu lực hiện tại.

    Args:
        law_name: Tên hoặc mô tả bộ luật cần tra cứu (ví dụ: 'Bộ luật Hình sự',
                  'Luật Phòng chống ma túy 2021', 'Luật 73/2021/QH14').
    """
    try:
        from weaviate.classes.query import Filter, MetadataQuery

        # Enrich query with effective-date keywords for better retrieval
        enriched_query = f"{law_name} hiệu lực thi hành ngày có hiệu lực"
        query_vector = _embed_query(enriched_query)
        client = _get_weaviate_client()
        collection = client.collections.get(WEAVIATE_COLLECTION)

        # Hybrid search filtered to legal documents only
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
        return "Không tìm thấy thông tin hiệu lực của bộ luật này trong cơ sở dữ liệu."
    except Exception as e:
        print(f"  ⚠ Weaviate unavailable ({e}), không thể tra cứu hiệu lực.")
        return "Không thể kết nối cơ sở dữ liệu pháp luật lúc này."


TOOLS = [
    search_legal_database,
    calculate_damages,
    check_statute_of_limitations,
    search_drug_case_news,
    check_law_effective_date,
]

QUESTION = "Luật Phòng chống ma túy 2025 có hiệu lực từ ngày nào? So với Luật 2021 thì có gì thay đổi?"


async def main():
    print("=" * 70)
    print("STAGE 2: LLM + RAG / Tools")
    print("=" * 70)
    print()
    print("[How it works]")
    print("  1. LLM receives tools (search_legal_database, calculate_damages,")
    print("     check_statute_of_limitations)")
    print("  2. search_legal_database queries Weaviate 'DrugLawDocs' collection")
    print("     via near_text (semantic) search for relevant legal chunks")
    print("  3. LLM decides which tools to call and with what arguments")
    print("  4. We execute the tools and feed results back to the LLM")
    print("  5. LLM generates a final answer grounded in retrieved data")
    print()
    print(f"Question: {QUESTION}")
    print("-" * 70)

    llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)
    tool_map = {t.name: t for t in TOOLS}

    messages = [
        SystemMessage(
            content=(
                "Bạn là chuyên gia pháp luật Việt Nam, đặc biệt về luật hình sự liên quan "
                "đến ma túy. Bạn có quyền truy cập vào cơ sở dữ liệu pháp luật (Weaviate "
                "vector store) và công cụ tính toán thiệt hại. Luôn sử dụng tool "
                "search_legal_database để tra cứu trước khi trả lời. "
                "Trả lời bằng tiếng Việt, trích dẫn điều luật cụ thể. "
                "Giữ câu trả lời dưới 400 từ."
            )
        ),
        HumanMessage(content=QUESTION),
    ]

    # --- Step 1: LLM decides which tools to call ---
    print("\n>>> Step 1: Asking LLM (with tools bound)...\n")
    response = await llm_with_tools.ainvoke(messages)
    messages.append(response)

    if not response.tool_calls:
        print("LLM chose not to use any tools. Direct answer:")
        print(response.content)
        return

    # --- Step 2: Execute tool calls ---
    print(f">>> Step 2: LLM requested {len(response.tool_calls)} tool call(s):\n")
    for tc in response.tool_calls:
        print(f"  Tool: {tc['name']}")
        print(f"  Args: {tc['args']}")

        tool_fn = tool_map[tc["name"]]
        result = await tool_fn.ainvoke(tc["args"])
        print(f"  Result: {result[:200]}{'...' if len(result) > 200 else ''}")
        print()

        messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    # --- Step 3: LLM generates final grounded answer ---
    print(">>> Step 3: LLM generating final answer with tool results...\n")
    final_response = await llm_with_tools.ainvoke(messages)
    print(final_response.content)

    print()
    print("-" * 70)
    print("[Improvements over Stage 1]")
    print("  + Grounded: answers cite specific Vietnamese drug-law statutes")
    print("  + RAG: Weaviate vector search retrieves relevant legal chunks")
    print("  + Tool use: can search databases and calculate damages")
    print("  + More accurate: retrieval reduces hallucination risk")
    print()
    print("[Limitations of Stage 2]")
    print("  - Manual orchestration: we wrote the tool-call loop ourselves")
    print("  - Single pass: only one round of tool calls")
    print("  - No reasoning loop: LLM can't decide to search again if needed")
    print()
    print("Next: Stage 3 wraps this in an autonomous ReAct agent loop.")
    print("=" * 70)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
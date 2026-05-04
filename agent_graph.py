import os
from typing import TypedDict, List, Dict, Literal
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
try:
    from langchain.tools.retriever import create_retriever_tool
except ImportError:
    from langchain_core.tools import create_retriever_tool

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langgraph.graph import StateGraph, END

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "vectorstore/faiss_index")

llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    temperature=0,
    groq_api_key=GROQ_API_KEY
)

fast_llm = ChatGroq(
    model=os.getenv("FAST_GROQ_MODEL", "llama-3.1-8b-instant"),
    temperature=0,
    groq_api_key=GROQ_API_KEY
)

embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
)


class OilGasState(TypedDict):
    user_query: str
    current_query: str
    retrieved_context: str
    evaluator_decision: str
    evaluator_feedback: str
    rewritten_query: str
    attempts: int
    final_answer: str
    report_plan: List[Dict[str, str]]
    current_section_index: int
    section_results: List[Dict[str, str]]
    final_report: str


def load_retriever():
    if not os.path.exists(VECTORSTORE_PATH):
        raise FileNotFoundError(
            f"FAISS vectorstore not found at {VECTORSTORE_PATH}. "
            "Run: python ingest.py"
        )

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore.as_retriever(
        search_kwargs={"k": int(os.getenv("RETRIEVER_K", "4"))}
    )


retriever = load_retriever()

oil_gas_retriever_tool = create_retriever_tool(
    retriever,
    name="internal_oil_gas_document_retriever",
    description=(
        "Use this tool first for questions about internal oil and gas field reports, "
        "drilling operations, production metrics, safety incidents, compliance controls, "
        "equipment downtime, methane monitoring, and Houston field operations."
    )
)


@tool
def tavily_web_search(query: str) -> str:
    """
    Search the web for external oil and gas information.
    Use this when internal documents are insufficient or when the question asks
    for current industry information, standards, or broader research.
    """
    search = TavilySearchResults(
        max_results=int(os.getenv("TAVILY_MAX_RESULTS", "4")),
        tavily_api_key=TAVILY_API_KEY
    )

    return str(search.invoke(query))


def retrieval_agent_node(state: OilGasState) -> Dict:
    query = state["current_query"]

    internal_docs = oil_gas_retriever_tool.invoke(query)

    web_context = ""

    web_decision_prompt = f"""
    You are deciding if web search is needed.

    User question:
    {query}

    Internal retrieved context:
    {internal_docs}

    Return only one word:
    YES_WEB or NO_WEB

    Use YES_WEB if:
    - internal context is empty
    - internal context is weak or unrelated
    - question asks for current industry information
    - question asks for external standards or broader research

    Use NO_WEB if:
    - internal context is enough to answer the question
    """

    web_decision = fast_llm.invoke(
        [HumanMessage(content=web_decision_prompt)]
    ).content.strip().upper()

    if "YES_WEB" in web_decision:
        web_context = tavily_web_search.invoke(query)

    combined_context = f"""
    INTERNAL DOCUMENT CONTEXT:
    {internal_docs}

    WEB SEARCH CONTEXT:
    {web_context}
    """

    return {
        "retrieved_context": combined_context,
        "attempts": state["attempts"] + 1
    }


def document_evaluator_node(state: OilGasState) -> Dict:
    prompt = f"""
    You are a retrieval quality evaluator for an Oil & Gas Operations Intelligence Agent.

    User question:
    {state['user_query']}

    Current retrieval query:
    {state['current_query']}

    Retrieved context:
    {state['retrieved_context']}

    Evaluate whether the retrieved context is sufficient to answer the user question.

    Return exactly this format:

    DECISION: YES or NO
    FEEDBACK: short reason
    REWRITTEN_QUERY: improved query if DECISION is NO, otherwise NONE

    Criteria:
    - Is the context directly relevant?
    - Does it include enough evidence?
    - Does it cover safety, production, drilling, compliance, or maintenance details if asked?
    - Is the answer grounded in internal documents or web context?
    """

    result = fast_llm.invoke([HumanMessage(content=prompt)]).content

    decision = "NO"
    if "DECISION: YES" in result.upper():
        decision = "YES"

    rewritten_query = state["current_query"]

    if "REWRITTEN_QUERY:" in result:
        rewritten_query = result.split("REWRITTEN_QUERY:")[-1].strip()

        if rewritten_query.upper() == "NONE":
            rewritten_query = state["current_query"]

    return {
        "evaluator_decision": decision,
        "evaluator_feedback": result,
        "rewritten_query": rewritten_query,
        "current_query": rewritten_query
    }


def answer_synthesizer_node(state: OilGasState) -> Dict:
    prompt = f"""
    You are an Oil & Gas Operations Intelligence Assistant.

    Guardrails:
    - Use only retrieved context.
    - Do not invent safety standards, production numbers, or incident details.
    - If evidence is insufficient, clearly say what is missing.
    - For safety questions, include risk controls and operational caution.
    - Do not expose sensitive or personal data.
    - Keep the answer professional and useful for internal operations teams.

    User question:
    {state['user_query']}

    Retrieved context:
    {state['retrieved_context']}

    Evaluator feedback:
    {state['evaluator_feedback']}

    Generate the final answer with:
    1. Direct Answer
    2. Supporting Evidence
    3. Operational Recommendation
    4. Safety/Compliance Note if relevant
    """

    result = llm.invoke([HumanMessage(content=prompt)]).content

    return {"final_answer": result}


def route_after_evaluation(state: OilGasState) -> Literal["retrieve_again", "answer"]:
    if state["evaluator_decision"] == "YES":
        return "answer"

    if state["attempts"] >= int(os.getenv("MAX_RETRIEVAL_ATTEMPTS", "3")):
        return "answer"

    return "retrieve_again"


def research_manager_node(state: OilGasState) -> Dict:
    topic = state["user_query"]

    planning_context = tavily_web_search.invoke(
        f"oil and gas research report structure for topic: {topic}"
    )

    prompt = f"""
    You are the Research Manager Agent for an internal Oil & Gas Operations Intelligence system.

    Broad research topic:
    {topic}

    Web context for planning only:
    {planning_context}

    Create a research plan for a structured report.

    Required final report structure:
    - Executive Summary
    - Key Findings
    - Detailed Analysis
    - Limitations and Further Research

    Your job:
    Create 3 to 5 Detailed Analysis sections.

    Each section should include:
    - section_title
    - research_question
    - what_to_research

    Do not perform the research yet.
    Return a simple numbered list.
    """

    result = llm.invoke([HumanMessage(content=prompt)]).content

    sections = []

    for line in result.splitlines():
        clean = line.strip("-• 0123456789.").strip()

        if clean and len(clean) > 20 and len(sections) < 5:
            sections.append({
                "section_title": clean[:90],
                "research_question": clean,
                "what_to_research": clean
            })

    if not sections:
        sections = [
            {
                "section_title": "Operational Overview",
                "research_question": f"What are the key operational trends related to {topic}?",
                "what_to_research": "Drilling, production, maintenance, and operational efficiency."
            },
            {
                "section_title": "Safety and Compliance",
                "research_question": f"What safety and compliance concerns are related to {topic}?",
                "what_to_research": "Safety risks, controls, standards, and compliance implications."
            },
            {
                "section_title": "Technology and Automation",
                "research_question": f"What technologies support {topic}?",
                "what_to_research": "AI, automation, monitoring, analytics, and optimization tools."
            }
        ]

    return {
        "report_plan": sections,
        "current_section_index": 0,
        "section_results": []
    }


def specialized_research_node(state: OilGasState) -> Dict:
    idx = state["current_section_index"]
    plan = state["report_plan"]

    if idx >= len(plan):
        return {}

    section = plan[idx]

    section_title = section.get("section_title", f"Section {idx + 1}")
    research_question = section.get("research_question", section_title)
    what_to_research = section.get("what_to_research", research_question)

    query_prompt = f"""
    Generate 2 concise web search queries for this oil and gas research section.

    Section title:
    {section_title}

    Research question:
    {research_question}

    What to research:
    {what_to_research}

    Return each query on a separate line.
    """

    query_text = fast_llm.invoke([HumanMessage(content=query_prompt)]).content

    queries = [
        q.strip("-• ").strip()
        for q in query_text.splitlines()
        if q.strip()
    ]

    queries = queries[:2] if queries else [research_question]

    web_results = []

    for q in queries:
        web_results.append({
            "query": q,
            "results": tavily_web_search.invoke(q)
        })

    summary_prompt = f"""
    You are a specialized oil and gas research agent.

    Section:
    {section_title}

    Research question:
    {research_question}

    Search results:
    {web_results}

    Summarize the findings for this section.

    Include:
    - Brief overview
    - Evidence-based findings
    - Operational relevance
    - Source references from the search results when available
    """

    section_summary = llm.invoke(
        [HumanMessage(content=summary_prompt)]
    ).content

    updated_results = state["section_results"] + [
        {
            "section_title": section_title,
            "research_question": research_question,
            "summary": section_summary
        }
    ]

    return {
        "section_results": updated_results,
        "current_section_index": idx + 1
    }


def route_research_progress(state: OilGasState) -> Literal["continue_research", "finalize_report"]:
    if state["current_section_index"] >= len(state["report_plan"]):
        return "finalize_report"

    return "continue_research"


def finalizer_node(state: OilGasState) -> Dict:
    prompt = f"""
    You are the finalizer for an Oil & Gas Deep Research report.

    Original topic:
    {state['user_query']}

    Completed section research:
    {state['section_results']}

    Generate a comprehensive report with exactly this structure:

    # Executive Summary

    # Key Findings

    # Detailed Analysis
    Include each researched section with supporting evidence.

    # Limitations and Further Research

    Guardrails:
    - Do not invent facts.
    - Clearly mention limitations.
    - Use professional enterprise language.
    - Focus on oil and gas operations, safety, compliance, production, and technology.
    """

    result = llm.invoke([HumanMessage(content=prompt)]).content

    return {"final_report": result}
def incident_root_cause_agent(incident: str) -> dict:
    """
    RAG-powered Incident Root Cause Analysis Agent.
    Uses internal oil & gas documents first, then generates grounded RCA.
    """

    internal_docs = oil_gas_retriever_tool.invoke(incident)

    prompt = f"""
You are an Oil & Gas Operations Root Cause Analysis Agent.

Use ONLY the internal document context below to analyze the incident.
If the context is not enough, clearly say what information is missing.
Do not invent exact facts, standards, readings, or historical incidents.

INTERNAL DOCUMENT CONTEXT:
{internal_docs}

INCIDENT:
{incident}

Return the answer in this exact structure:

1. Incident Summary
2. Internal Knowledge Used
3. Possible Root Causes
4. Risk Level: Low / Medium / High / Critical
5. Immediate Recommended Actions
6. Preventive Measures
7. Missing Information Needed
8. Safety / Compliance Notes
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "incident": incident,
        "retrieved_context": internal_docs,
        "analysis": response.content
    }

# -----------------------------
# Q&A Agentic RAG Graph
# -----------------------------

qa_builder = StateGraph(OilGasState)

qa_builder.add_node("retrieval_agent", retrieval_agent_node)
qa_builder.add_node("document_evaluator", document_evaluator_node)
qa_builder.add_node("answer_synthesizer", answer_synthesizer_node)

qa_builder.set_entry_point("retrieval_agent")

qa_builder.add_edge("retrieval_agent", "document_evaluator")

qa_builder.add_conditional_edges(
    "document_evaluator",
    route_after_evaluation,
    {
        "retrieve_again": "retrieval_agent",
        "answer": "answer_synthesizer"
    }
)

qa_builder.add_edge("answer_synthesizer", END)

qa_graph = qa_builder.compile()


# -----------------------------
# Deep Research Graph
# -----------------------------

research_builder = StateGraph(OilGasState)

research_builder.add_node("research_manager", research_manager_node)
research_builder.add_node("specialized_researcher", specialized_research_node)
research_builder.add_node("finalizer", finalizer_node)

research_builder.set_entry_point("research_manager")

research_builder.add_edge("research_manager", "specialized_researcher")

research_builder.add_conditional_edges(
    "specialized_researcher",
    route_research_progress,
    {
        "continue_research": "specialized_researcher",
        "finalize_report": "finalizer"
    }
)

research_builder.add_edge("finalizer", END)

research_graph = research_builder.compile()


def _initial_state(input_text: str) -> OilGasState:
    return {
        "user_query": input_text,
        "current_query": input_text,
        "retrieved_context": "",
        "evaluator_decision": "",
        "evaluator_feedback": "",
        "rewritten_query": "",
        "attempts": 0,
        "final_answer": "",
        "report_plan": [],
        "current_section_index": 0,
        "section_results": [],
        "final_report": ""
    }


def run_qa(question: str):
    return qa_graph.invoke(
        _initial_state(question),
        config={
            "run_name": "oil-gas-agentic-rag-qa",
            "tags": ["oil-gas", "agentic-rag", "qa"]
        }
    )


def run_deep_research(topic: str):
    return research_graph.invoke(
        _initial_state(topic),
        config={
            "run_name": "oil-gas-deep-research",
            "tags": ["oil-gas", "deep-research", "report"]
        }
    )
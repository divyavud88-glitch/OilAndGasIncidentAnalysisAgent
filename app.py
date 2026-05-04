from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent_graph import run_qa, run_deep_research, incident_root_cause_agent

app = FastAPI(
    title="Oil & Gas Operations Intelligence Agent",
    description="Production-style Agentic RAG and Deep Research system for oil and gas operations.",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    question: str


class ResearchRequest(BaseModel):
    topic: str

class IncidentRequest(BaseModel):
    incident: str

@app.get("/")
def home():
    return {"message": "Oil & Gas Operations Intelligence Agent is running"}

@app.get("/health")
def health():
    return {"status": "ok", "service": "oil-gas-operations-intelligence-agent"}


@app.post("/ask")
def ask_agent(request: QueryRequest):
    try:
        result = run_qa(request.question)
        return {
            "question": request.question,
            "answer": result.get("final_answer"),
            "evaluator_decision": result.get("evaluator_decision"),
            "evaluator_feedback": result.get("evaluator_feedback"),
            "attempts": result.get("attempts")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deep-research")
def deep_research(request: ResearchRequest):
    try:
        result = run_deep_research(request.topic)
        return {
            "topic": request.topic,
            "report_plan": result.get("report_plan"),
            "section_results": result.get("section_results"),
            "final_report": result.get("final_report")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/incident-analysis")
def incident_analysis(request: IncidentRequest):
    """
    Incident Root Cause Analysis Agent endpoint.
    """

    result = incident_root_cause_agent(request.incident)

    return {
        "status": "success",
        "agent": "Incident Root Cause Analysis Agent",
        "result": result
    }
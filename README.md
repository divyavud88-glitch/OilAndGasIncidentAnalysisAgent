# 🛢️ Oil & Gas Operations Intelligence Agent

## 🚀 Overview

The **Oil & Gas Operations Intelligence Agent** is a production-style, AI-powered system designed to assist engineers and operators in analyzing operational issues, retrieving domain knowledge, and generating actionable insights.

This system leverages **Agentic RAG (Retrieval-Augmented Generation)**, **multi-agent workflows**, and **LLMs** to provide grounded, context-aware responses for oil & gas operations.

---

## 🎯 Problem Statement

Oil & gas operations involve complex systems where failures can lead to:

* ⚠️ Safety risks (leaks, pressure failures, explosions)
* 💰 High operational costs due to downtime
* 🧠 Dependency on experienced engineers for troubleshooting
* 📄 Time-consuming manual analysis of documents and reports

### ❌ Traditional Approach

* Manual troubleshooting
* Searching through documents
* Delayed decision-making

### ✅ Solution

This project introduces an **AI-driven agent system** that:

* Diagnoses operational incidents
* Retrieves internal engineering knowledge
* Combines internal data with external insights
* Generates structured, actionable responses

---

## 🧠 Key Features

### 🔹 1. Agentic RAG Q&A System (`/ask`)

* Retrieves relevant context from internal documents (FAISS)
* Dynamically decides whether to use:

  * Internal knowledge (RAG)
  * External web search (Tavily)
* Uses multi-step reasoning with evaluation loop

---

### 🔹 2. Incident Root Cause Analysis Agent (`/incident-analysis`)

* Analyzes operational issues such as:

  * Pressure drops
  * Pipeline leaks
  * Pump failures
  * Compressor abnormalities
* Uses **RAG-powered grounding** from internal documents
* Generates:

  * Root causes
  * Risk levels
  * Immediate actions
  * Preventive measures
  * Safety/compliance notes

---

### 🔹 3. Deep Research Agent (`/deep-research`)

* Performs multi-step research using:

  * Web search (Tavily)
  * Section-wise analysis
* Generates structured reports:

  * Executive Summary
  * Key Findings
  * Detailed Analysis
  * Limitations

---

### 🔹 4. Intelligent Tool Routing

* Decides when to use:

  * RAG (internal FAISS vector DB)
  * Web search (Tavily)
* Implements **Agentic decision-making**

---

### 🔹 5. Retrieval Evaluation Loop

* Evaluates quality of retrieved context
* Rewrites query if needed
* Improves answer accuracy

---

## 🏗️ Architecture

```text
User Query
   ↓
Agent Router
   ↓
-----------------------------------------
| Internal RAG | Web Search | Analysis   |
|   (FAISS)    | (Tavily)   |  Agent     |
-----------------------------------------
   ↓
LLM Reasoning (Groq - LLaMA Models)
   ↓
Final Structured Response
```

---

## 🧰 Tech Stack

### 🔹 Core AI & LLM

* LangChain
* LangGraph
* Groq (LLaMA models)
* HuggingFace Embeddings

### 🔹 Retrieval

* FAISS Vector Database

### 🔹 Tools

* Tavily Web Search API

### 🔹 Backend

* FastAPI
* Uvicorn

### 🔹 Observability

* LangSmith (tracing & debugging)

### 🔹 Environment

* Python 3.10
* dotenv

---

## 📡 API Endpoints

### ✅ `/ask`

General oil & gas Q&A using Agentic RAG

### ✅ `/incident-analysis`

Root cause analysis for operational incidents

### ✅ `/deep-research`

Generates structured research reports

### ✅ `/health`

Service health check

---

## 🧪 Example Use Case

### Input:

```json
{
  "incident": "Pipeline pressure dropped suddenly during operation"
}
```

### Output:

* Root cause analysis
* Risk level
* Recommended actions
* Preventive measures

---

## 📊 Observability (LangSmith)

This project integrates **LangSmith** for:

* Tracing agent execution
* Debugging tool usage
* Monitoring LLM performance

📸 You can include screenshots of:

* Agent traces
* Tool calls
* Latency breakdown

---

## ⚙️ Setup Instructions

```bash
git clone <your-repo-url>
cd oil-gas-operations-intelligence-agent

python -m venv venv310
venv310\Scripts\activate

pip install -r requirements.txt
```

### Run the application

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

## 🔐 Environment Variables (.env)

```env
GROQ_API_KEY=your_key
TAVILY_API_KEY=your_key
VECTORSTORE_PATH=vectorstore/faiss_index
```

---

## 🚀 Future Enhancements

* Redis-based semantic caching
* PDF upload + document ingestion
* Real-time monitoring dashboards
* Multi-agent orchestration with LangGraph
* Deployment using Docker & AWS

---

## 💼 Why This Project Matters

This project demonstrates:

* Real-world **AI application in Oil & Gas domain**
* **Agentic RAG architecture**
* **Production-style system design**
* Integration of **LLMs, retrieval, and APIs**
* Strong alignment with **industry use cases (Houston Energy sector)**

---

## 🙌 Author

**Divya E**
AI / Agentic AI Developer

---

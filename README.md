# Legal Contract Review Agent

An AI-powered agentic pipeline that automates end-to-end vendor contract review. Built with LangGraph, the agent extracts every clause, scores risk against a RAG knowledge base of standard templates, drafts professional legal findings, and generates two structured PDF reports — one for the reviewing lawyer and one for the client.

---

## What it does

- Reads a raw PDF contract and classifies it (Consulting, SaaS, Employment, VendorSupply, DataProcessing)
- Extracts every numbered sub-clause using an LLM with deduplication
- Retrieves relevant reference chunks from a FAISS vector store for each clause
- Classifies each clause: type, risk level (low/medium/high/critical), deviation, and recommended action
- Drafts a professional legal finding per clause
- Generates a **Draft PDF** for the lawyer to review
- Pauses for **human-in-the-loop** lawyer review — lawyer can modify specific clause recommendations
- Rewrites modified findings using the LLM based on the lawyer's action and reason
- Generates a **Final PDF** stamped APPROVED BY LEGAL or RETURNED FOR RENEGOTIATION

---

## Tech Stack

| Component | Technology |
|---|---|
| Agentic Framework | LangGraph (StateGraph + InMemorySaver) |
| LLM | Groq — LLaMA 3.3 70B Versatile |
| Embeddings | Ollama nomic-embed-text |
| Vector Store | FAISS |
| PDF Generation | ReportLab |
| Document Loading | pypdf |
| Tracing | LangSmith |
| Environment | Jupyter Notebook |

---

## Pipeline

```
START → ReadContract → ClassifyContract → ExtractClauses → SearchRAG → CheckIssues → DraftFinding → GenerateReport → LawyerReview → FinaliseReport → END
```

| Node | Description |
|---|---|
| ReadContract | Validates contract text, generates Ticket ID and Review ID |
| ClassifyContract | Samples start/middle/end of contract and classifies type |
| ExtractClauses | LLM extracts all numbered sub-clauses with deduplication |
| SearchRAG | Retrieves top 3 reference chunks per clause from FAISS |
| CheckIssues | Classifies each clause: type, risk, deviation, recommendation |
| DraftFinding | Writes a 3-4 sentence legal finding per clause |
| GenerateReport | Generates plain Draft PDF for lawyer (no LLM call) |
| LawyerReview | Pauses graph via interrupt() for human review and approval |
| FinaliseReport | Applies lawyer changes, rewrites findings, generates Final PDF |

---

## Output

**Draft PDF** (`Draft_Review_{TicketId}.pdf`)
- Plain black and white
- Stamped: DRAFT — FOR LEGAL REVIEW ONLY
- Risk summary table + all clause findings with original text, deviation, recommendation, and draft finding

**Final PDF** (`Final_Review_{TicketId}.pdf`)
- Colour-coded risk banner (Critical / High / Medium / Low)
- Executive Summary: The Bottom Line, Key Risk Areas, Next Steps
- Clause-by-Clause Findings with The Issue / The Impact / Action Required
- Lawyer-modified clauses marked `[Updated per legal review]`
- Stamped: APPROVED BY LEGAL or RETURNED FOR RENEGOTIATION

---

## Risk Classification

| Level | Action | Condition |
|---|---|---|
| CRITICAL | Reject / Escalate | Removes non-negotiable client protection or creates uncapped obligation |
| HIGH | Negotiate | Significant vendor advantage or restricts client ability to exit or seek remedy |
| MEDIUM | Flag for Review | Deviation from standard but client retains partial protections |
| LOW | Accept | Consistent with reference standard or only marginally different |

---

## Project Structure

```
legal-contract-review-agent/
│
├── main.ipynb                          # Main notebook — full pipeline
├── README.md                           # This file
│
├── data/
│   ├── standard_templates.pdf          # Balanced contract reference templates
│   ├── vendor_contracts.pdf            # Vendor-friendly risky contract examples
│   └── risk_guidelines.pdf             # Risk classification rules and red flags
│
├── outputs/
│   ├── Draft_Review_{TicketId}.pdf     # Lawyer draft report
│   └── Final_Review_{TicketId}.pdf     # Final client report
│
└── docs/
    └── Legal_Contract_Review_Agent_Documentation.docx
```

---

## Setup

### Prerequisites

```bash
pip install langgraph langchain langchain-groq langchain-community faiss-cpu pypdf reportlab
```

```bash
ollama pull nomic-embed-text
```

### API Keys

```python
import os
import getpass

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass("LangSmith API Key: ")
os.environ["GROQ_API_KEY"]      = getpass.getpass("Groq API Key: ")
```

### LLM Initialisation

```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.environ["GROQ_API_KEY"]
)
```

### RAG Corpus Setup

Place your PDF documents in the `data/` folder and run the RAG setup cell in the notebook. The cell loads all three PDFs, chunks them, embeds them using nomic-embed-text, and builds the FAISS index.

> **Note:** The FAISS index is not persistent — it must be rebuilt on each kernel restart.

---

## Running the Pipeline

Open `main.ipynb` and run all cells in order. The invocation cell at the bottom handles the two-phase flow.

### Phase 1

```python
InitialState = {
    "contract_text": ContractText,
    "contract_name": ContractName,
    "client_name":   ClientName,
}

ThreadId = str(uuid.uuid4())
Config   = {"configurable": {"thread_id": ThreadId}}

App.invoke(InitialState, Config)
```

Runs ReadContract through FinaliseReport, pauses at `interrupt()` inside LawyerReview. The Draft PDF is saved at this point.

### Lawyer Review Panel

```
==================================================
LAWYER REVIEW PANEL
==================================================
Contract:      CONSULTING SERVICES AGREEMENT
Client:        ClientCo Ltd.
Ticket:        LEG-XXXXXXXX
Total Clauses: 27
Critical: 5 | High: 15 | Medium: 6 | Low: 1
==================================================

High and Critical Clauses:

  Clause 10: IP_OWNERSHIP
    Risk:           CRITICAL
    Action:         Reject
    Recommendation: Reject this clause and negotiate for client ownership of all deliverables

  Clause 20: LIABILITY_CAP
    Risk:           CRITICAL
    Action:         Reject
    Recommendation: Reject and negotiate for a liability cap tied to fees paid
  ...

How many clauses to modify? 1

  Change 1 of 1:
  Clause number: 20
  Updated action required: Negotiate minimum 6 months fees as liability floor
  Reason for change: $500 cap is commercially unenforceable, fee-based floor is standard

Approve report for client delivery? (yes/no): yes
```

### Phase 2

```python
FinalResult = App.invoke(Command(resume=LawyerPayload), Config)
```

Resumes the graph. FinaliseReport rewrites the modified clause finding, generates the Final PDF, and returns `status: Complete`.

---

## State Schema

```python
class ContractReviewState(TypedDict):
    contract_text:     str
    contract_name:     str
    client_name:       str
    review_id:         str
    contract_type:     Literal["consulting", "saas", "employment",
                               "vendorsupply", "dataprocessing", "other"] | None
    extracted_clauses: list[str] | None
    classifications:   list[ContractClassification] | None
    retrieved_context: list[str] | None
    draft_findings:    list[str] | None
    final_report:      str | None
    needs_human_review: bool
    human_feedback:    dict | None
    ticket_id:         str | None
    status:            Literal["Pending", "InReview", "Escalated", "Complete"] | None
```

```python
class ContractClassification(TypedDict):
    RiskLevel:      Literal["low", "medium", "high", "critical"]
    ClauseType:     Literal["payment_terms", "termination", "liability_cap",
                            "data_ownership", "auto_renewal", "ip_ownership",
                            "non_compete", "indemnification",
                            "breach_notification", "other"]
    ActionRequired: Literal["Reject", "Negotiate", "Flag", "Accept"]
    Deviation:      str
    Recommendation: str
```

---

## Key Design Decisions

**LLM choice** — Groq LLaMA 3.3 70B over local Ollama. Larger context window (8K tokens) handles full contracts without truncation. Significantly faster inference for 27+ clause pipeline runs.

**Clause deduplication** — Word overlap similarity check at 80% threshold removes near-duplicate clauses that LLMs tend to extract from the same contract section, reducing noise in the report.

**Start/middle/end sampling** — ClassifyContract samples 1000 + 1000 + 500 characters from the start, midpoint, and end of the contract rather than the first 2000 characters. The middle and end sections contain the most contract-type-specific language.

**No LLM in GenerateReport** — The Draft PDF is built entirely from state data. This saves tokens and guarantees the lawyer always receives a report even if the Groq quota is exhausted mid-run.

**Programmatic Executive Summary** — The Bottom Line, Key Risk Areas, and Next Steps in the Final PDF are generated programmatically from state data, not by the LLM. This eliminates hallucination risk in the report output.

**Rate limiting** — `time.sleep(5)` between LLM calls in both CheckIssues and DraftFinding prevents hitting Groq free tier token-per-minute limits during long pipeline runs.

**Single interrupt** — All `input()` prompts are collected in the invocation cell before resuming the graph. This avoids the LangGraph behaviour where re-running a node on resume re-fires all input prompts, which would cause the lawyer to enter data twice.

---

## Known Limitations

- **Groq free tier:** 100K tokens per day. A full 27-clause pipeline consumes approximately 60–70K tokens, allowing roughly 1–2 full runs per day
- **Context window:** LLaMA 3.3 70B has an 8K token limit. Very long contracts (20+ pages) may require chunked extraction
- **Single contract per run:** Batch processing requires a loop over the invocation cell with a fresh `ThreadId` per contract
- **FAISS persistence:** The vector store is rebuilt on each kernel restart — not persistent between sessions

---

## Code Conventions

- PascalCase for all variable names throughout the notebook
- No emojis in any code cells
- Relative paths only — no hardcoded absolute paths
- `OllamaEmbeddings` retained for FAISS; `ChatOllama` replaced by `ChatGroq`
- `interrupt()` used only inside `LawyerReview`
- `checkpointer=Memory` always present in `Builder.compile()`
- Markdown fence handling: `split("```")` then strip `json` prefix
- All `except` blocks catch `Exception`, not just `JSONDecodeError`

---

## Project Context

Built as a portfolio project during a Gen AI internship at a consulting firm. Demonstrates enterprise-grade agentic AI competencies including multi-node LangGraph pipelines, RAG, human-in-the-loop design, structured JSON prompting, and production PDF generation — skills directly applicable to AI consulting engagements where contract review is a common and expensive manual process.

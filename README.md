# 🏥 MedRAG — Agentic Medical AI Assistant

> An AI-powered medical assistant that combines **evidence-based RAG** with a **LangGraph agentic pipeline** — answering clinical questions from trusted documents *and* booking specialist consultations, all through a single premium Streamlit UI.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🧠 **Agentic Orchestration** | LangGraph `StateGraph` with intent routing, multi-turn slot collection, and human-in-the-loop approval |
| 🔍 **Semantic RAG** | MongoDB Atlas Vector Search + HuggingFace MiniLM embeddings |
| ⚡ **Groq Cloud LLM** | `llama-3.3-70b-versatile` via Groq API — blazing fast inference |
| 📅 **Doctor Booking** | Multi-turn slot collection → availability check → HITL approval → MongoDB write |
| 🩺 **Smart Consult Offer** | LLM classifies RAG answers — only offers specialist booking when clinically warranted |
| 🔄 **Reranking** | Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) for improved retrieval precision |
| 🚨 **Emergency Detection** | Regex-based fast-path — detects acute emergencies and redirects immediately |
| 📄 **Multi-format Ingestion** | PDF, TXT, JSON/JSONL, CSV — drag-and-drop via UI or CLI |
| ⚕️ **Medical Disclaimer** | Automatically appended to every generated response |
| 🔗 **Source Citations** | Every answer cites the exact document and chunk it was drawn from |
| ✅ **Grounding Check** | Flags responses with low lexical overlap with retrieved context |
| 💬 **Premium Chat UI** | Dark-mode Streamlit interface with animations, streaming, and appointment history |

---

## 🏗️ Architecture

```
User Query (Streamlit UI)
         │
         ▼
┌────────────────────────────────────────────┐
│          LangGraph StateGraph              │
│                                            │
│  ┌─────────────────────────────────────┐   │
│  │       Intent Router Node            │   │
│  │  • Regex → emergency / off-topic    │   │
│  │  • Groq LLM → clinical | booking    │   │
│  └───────────┬──────────────┬──────────┘   │
│              │              │               │
│        clinical          booking            │
│              │              │               │
│              ▼              ▼               │
│  ┌─────────────────┐  ┌──────────────────┐ │
│  │   RAG Node      │  │  Slot Collector  │ │
│  │ • Retrieve      │  │  • Extract name  │ │
│  │ • Rerank        │  │  • Specialty     │ │
│  │ • Generate      │  │  • Preferred date│ │
│  │ • Offer check   │  └────────┬─────────┘ │
│  └────────┬────────┘           │            │
│           │             ┌──────▼──────────┐ │
│           │             │ Availability    │ │
│           │             │ Checker Node    │ │
│           │             │ (Mock Calendar) │ │
│           │             └──────┬──────────┘ │
│           │                    │            │
│           │          ┌─────────▼──────────┐ │
│           │          │ Human-in-the-Loop  │◄── INTERRUPT
│           │          │  Slot Review Card  │ │
│           │          └─────────┬──────────┘ │
│           │                    │            │
│           │          ┌─────────▼──────────┐ │
│           │          │   DB Writer Node   │ │
│           │          │ MongoDB appointments│ │
│           │          └────────────────────┘ │
└────────────────────────────────────────────┘
         │
         ▼
   Final Response
 + Source Citations
 + Appointment Card
```

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | [python.org](https://www.python.org/downloads/) |
| Groq API Key | Free at [console.groq.com](https://console.groq.com) |
| MongoDB Atlas account | [cloud.mongodb.com](https://cloud.mongodb.com) — free tier works |

> **No Ollama required.** MedRAG v2 uses the Groq Cloud API for LLM inference.

---

### 1. Clone & Set Up Environment

```bash
cd MedRag

# Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Configure `.env`

Copy the example and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# MongoDB Atlas
MONGODB_URI=mongodb+srv://<user>:<password>@cluster0.xxxxx.mongodb.net/

# Groq Cloud API  →  get free key at https://console.groq.com
GROQ_API_KEY=gsk_...

# Model — use one of the available models for your API key tier
GROQ_MODEL=openai/gpt-oss-120b

# Calendar mode: "mock" uses MongoDB doctor roster; "google" = real Calendar API
CALENDAR_MODE=mock
```

---

### 3. Set Up MongoDB Atlas

1. Create a free cluster at [cloud.mongodb.com](https://cloud.mongodb.com).
2. In **Network Access**, allow your IP address.
3. In **Database Access**, create a user with read/write permissions.
4. Copy your connection string into `MONGODB_URI`.

**Create the Vector Search Index** (after first ingestion):

1. Atlas UI → your cluster → **Atlas Search** → **Create Search Index**
2. Select **Atlas Vector Search** → **JSON Editor**
3. Database: `medrag` · Collection: `medical_chunks`
4. Paste this definition:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 384,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "source"
    }
  ]
}
```

5. Name the index `medrag_vector_index` → **Create**.

---

### 4. Add Medical Documents

Place documents in `data/documents/`:

```
MedRag/
└── data/
    └── documents/
        ├── diabetes_guidelines.pdf
        ├── hypertension_facts.txt
        ├── pubmed_abstracts.jsonl
        └── drug_interactions.csv
```

> 💡 Free public sources: [CDC](https://www.cdc.gov), [WHO Publications](https://www.who.int/publications), [MedlinePlus](https://medlineplus.gov)

---

### 5. Ingest Documents

```bash
# Ingest all documents in data/documents/
python -m src.ingestion

# Fresh start (wipes existing chunks):
python -m src.ingestion --reset

# Custom directory:
python -m src.ingestion --docs-dir ./my_docs
```

---

### 6. Launch the App

```bash
streamlit run src/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

**First-time setup in the UI:**
- Click **🌱 Seed Doctor Roster** in the sidebar to populate the doctors collection (needed for booking).

---

## 🤖 Agentic Flows

### Clinical Q&A Flow

1. Ask any medical question → intent router classifies as **`clinical`**
2. RAG node retrieves top-K chunks from MongoDB Atlas, reranks, and generates an answer via Groq
3. If the answer describes a specific condition warranting specialist care → **"Would you like to consult a specialist?"** banner appears with a specialty selector

### Doctor Booking Flow

Triggered by either:
- Clicking **📅 Book Appointment** after a clinical answer
- Typing a booking intent directly (e.g. *"Book a cardiologist appointment"*)

```
Booking Intent Detected
        │
        ▼
Slot Collector (multi-turn)
  Q: "What is the patient's full name?"
  Q: "Which specialist?" → validates against allowed list
  Q: "What date?" → converts relative dates (e.g. "next Monday")
        │
        ▼
Availability Checker
  → Queries MongoDB doctors collection for the specialty + weekday
  → Returns available (doctor, date, time) slots
        │
        ▼
Human-in-the-Loop Card ← GRAPH PAUSES HERE
  → User selects a slot via radio buttons
  → Clicks ✅ Confirm or ❌ Cancel
        │
        ▼
DB Writer → MongoDB `appointments` collection
  → Confirmation card with booking ID shown in chat
  → Appointment appears in sidebar "📋 Recent Appointments"
```

---

## 📁 Project Structure

```
MedRag/
├── .env                          # Local config (not committed)
├── .env.example                  # Environment variable template
├── requirements.txt              # Python dependencies
├── README.md
├── data/
│   └── documents/                # Place your medical PDFs/TXTs here
└── src/
    ├── __init__.py
    ├── app.py                    # Streamlit chat UI (agentic edition)
    ├── config.py                 # Centralised config (env vars + constants)
    ├── generator.py              # Groq prompt construction + streaming generation
    ├── guardrails.py             # Emergency detection, grounding check, disclaimer
    ├── ingestion.py              # Document loading, chunking, embedding, MongoDB upsert
    ├── retriever.py              # Query embedding + vector search + cross-encoder reranking
    ├── utils.py                  # Logging, text helpers
    └── agent/                    # LangGraph agentic package
        ├── __init__.py
        ├── state.py              # AgentState TypedDict (shared graph state)
        ├── graph.py              # StateGraph assembly, routing, INTERRUPT logic
        ├── intent_router.py      # Intent classification node
        ├── rag_node.py           # RAG pipeline node + specialist-offer classifier
        ├── slot_collector.py     # Multi-turn booking form slot extractor
        ├── calendar_tool.py      # Mock calendar (MongoDB doctors collection)
        ├── db_writer.py          # Appointment persistence node
        └── seed_doctors.py       # One-shot doctor roster seeder
```

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `MONGODB_URI` | *(required)* | MongoDB Atlas connection string |
| `MONGODB_DB_NAME` | `medrag` | Database name |
| `MONGODB_COLLECTION_NAME` | `medical_chunks` | RAG chunks collection |
| `MONGODB_VECTOR_INDEX` | `medrag_vector_index` | Atlas Vector Search index name |
| `APPOINTMENTS_COLLECTION` | `appointments` | Confirmed bookings collection |
| `DOCTORS_COLLECTION` | `doctors` | Doctor roster collection (mock calendar) |
| `GROQ_API_KEY` | *(required)* | Groq Cloud API key |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Active models: `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`, `allam-2-7b` |
| `CALENDAR_MODE` | `mock` | `mock` = MongoDB roster · `google` = Google Calendar API |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `DOCS_DIR` | `./data/documents` | Source documents directory |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between adjacent chunks |
| `TOP_K` | `5` | Chunks to retrieve per query |

---

## 🧪 Testing Modules Individually

```bash
# Test retrieval pipeline
python -m src.retriever "What are symptoms of hypertension?"

# Test end-to-end RAG generation (Groq)
python -m src.generator "How is type 2 diabetes managed?"

# Test the full agentic graph (clinical query)
python -m src.agent.graph "What causes chronic kidney disease?"

# Seed the doctor roster manually
python -m src.agent.seed_doctors
```

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| **LLM** | Groq Cloud API — `llama-3.3-70b-versatile` |
| **Agentic Orchestration** | LangGraph `StateGraph` with `MemorySaver` checkpointer |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` (local, CPU) |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| **Vector Database** | MongoDB Atlas Vector Search |
| **Appointment DB** | MongoDB Atlas (`appointments` collection) |
| **UI Framework** | Streamlit 1.44 |
| **LLM Framework** | LangChain + LangGraph |

---

## ⚕️ Medical Disclaimer

**This application is for educational and informational purposes only.** It does not provide medical advice, diagnosis, or treatment recommendations. Always consult a qualified healthcare professional for medical concerns.

In case of a medical emergency, call emergency services immediately (911, 999, or your local equivalent).

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

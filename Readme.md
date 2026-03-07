# 🧮 Math Tutor Agent

An AI-powered JEE mathematics tutor built with **LangGraph**, **Streamlit**, and **Groq (LLaMA 3.3 70B)**. The system accepts text, image (OCR), or audio (ASR) input, solves problems step-by-step using a multi-agent ReAct pipeline, verifies its own answers, generates rich explanations, and can render animated **Manim visualisations** via a local MCP server — all with human-in-the-loop checkpoints at every ambiguous decision.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Multi-modal input** | Text, image (OCR), or audio (ASR) |
| **Multi-agent pipeline** | Parser → Intent Router → Solver → Verifier → Explainer |
| **ReAct tool loop** | RAG (PDF), Web Search (DuckDuckGo), Python Calculator |
| **RAG over student PDFs** | Cohere `embed-english-v3.0` + FAISS `IndexFlatIP` (cosine similarity) |
| **Human-in-the-Loop** | Clarification HITL (ambiguous problems) + Satisfaction HITL (post-answer) |
| **Manim visualisation** | Auto-generated animations via local FastMCP server |
| **Conversation threads** | LangGraph `InMemorySaver` checkpointing per thread |
| **Activity panel** | Live agent activity sidebar showing every node decision |

---

## 🏗️ Architecture

```mermaid
flowchart TD
    %% ── Input Layer ────────────────────────────────────────────
    USER(["👤 User\n(Text / Image / Audio)"])
    APP["🖥️ Streamlit Frontend\nnew_app.py"]

    USER -->|"question"| APP

    %% ── LangGraph Entry ────────────────────────────────────────
    APP -->|"stream / Command(resume)"| DETECT

    subgraph LANGGRAPH["🔁 LangGraph StateGraph  •  AgentState  •  InMemorySaver"]
        direction TB

        DETECT["🔍 detect_input\nClassify: text / image / audio"]

        OCR["📷 ocr_node\nMediaProcessor → text + confidence"]
        ASR["🎙️ asr_node\nMediaProcessor → transcript + confidence"]

        PARSER["📝 parser_agent\nLLaMA-3.3-70B structured output\n→ ParserOutput\n(clean text, topic, vars, constraints)"]

        HITL_C["🙋 hitl_node\ninterrupt() — waits for\nhuman clarification\nResumed via Command(resume=)"]

        ROUTER["🧭 intent_router\nLLaMA-3.3-70B structured output\n→ IntentRouterOutput\n(topic, difficulty, strategy,\nneeds_visualization, viz_hint)"]

        subgraph REACT["⚙️ ReAct Tool Loop"]
            direction LR
            SOLVER["🧠 solver_agent\nLLaMA-3.3-70B bind_tools\ntool_choice=auto\n→ SolverOutput"]
            TOOLS["🔧 tool_node\n• rag_tool\n• web_search_tool\n• python_calculator_tool"]
            SOLVER -- "tool_calls?" --> TOOLS
            TOOLS -- "ToolMessage" --> SOLVER
        end

        VERIFIER["✅ verifier_agent\nLLaMA-3.3-70B structured output\n→ VerifierOutput\n(correct / incorrect / needs_human)"]

        EXPLAINER["📘 explainer_agent\nCall 1: structured ExplainerOutput\n(steps, key concepts, common mistakes)\nCall 2: plain text Manim code\n(separate call — avoids Groq 400)"]

        MANIM["🎬 manim_node\nFastMCP client → local Manim server\nasyncio + nest_asyncio\n→ renders .mp4"]

        HITL_S["✅ satisfaction_hitl\ninterrupt() — asks user\nif satisfied with explanation\nResumed via Command(resume=)"]

        %% ── Flow ───────────────────────────────────────────────
        DETECT -->|"image"| OCR
        DETECT -->|"audio"| ASR
        DETECT -->|"text"| PARSER
        DETECT -->|"no input"| HITL_C
        OCR --> PARSER
        ASR --> PARSER
        PARSER -->|"needs_clarification"| HITL_C
        PARSER -->|"clear"| ROUTER
        HITL_C -->|"no solver_output yet"| PARSER
        HITL_C -->|"iteration cap hit"| PARSER
        HITL_C -->|"mid-solve needs_human"| SOLVER
        ROUTER --> SOLVER
        SOLVER -->|"no tool_calls → done"| VERIFIER
        VERIFIER -->|"correct"| EXPLAINER
        VERIFIER -->|"incorrect / partial\niter < MAX"| SOLVER
        VERIFIER -->|"needs_human"| HITL_C
        EXPLAINER --> MANIM
        MANIM --> HITL_S
        HITL_S -->|"satisfied"| END_NODE(["🏁 END"])
        HITL_S -->|"not satisfied"| PARSER
    end

    %% ── Tools Detail ───────────────────────────────────────────
    subgraph TOOLS_DETAIL["🔧 Solver Tools"]
        direction TB
        RAG["📄 rag_tool\nCohere embed-english-v3.0\nFAISS IndexFlatIP\nCosine similarity ≥ 0.30\nTOP_K = 5 chunks"]
        WEB["🌐 web_search_tool\nDuckDuckGoSearchRun"]
        CALC["🐍 python_calculator_tool\nSafe Python eval sandbox\nmath / cmath built-ins\nno numpy — pure Python"]
    end

    TOOLS -.->|"calls"| RAG
    TOOLS -.->|"calls"| WEB
    TOOLS -.->|"calls"| CALC

    %% ── MCP Server ─────────────────────────────────────────────
    subgraph MCP["🖥️ Local MCP Server\nmanim_mcp_server.py\nlocalhost:8765/mcp"]
        MANIM_SERVER["Manim CE\nrender_scene(code, class)\n→ .mp4 output path"]
    end

    MANIM <-->|"FastMCP async\ncall_tool(render_scene)"| MCP

    %% ── Output ─────────────────────────────────────────────────
    LANGGRAPH -->|"stream_mode=updates\nfinal_response"| APP
    APP -->|"st.write_stream\nst.video\nactivity panel"| USER

    %% ── Styling ────────────────────────────────────────────────
    classDef agent fill:#1e3a5f,stroke:#4a9eff,color:#e8f4fd
    classDef hitl fill:#5c2a00,stroke:#ff8c42,color:#fff0e0
    classDef tool fill:#1a3a1a,stroke:#4caf50,color:#e8fce8
    classDef io fill:#2a1a3a,stroke:#9c27b0,color:#f3e5f5
    classDef server fill:#3a1a1a,stroke:#f44336,color:#fce8e8

    class DETECT,OCR,ASR,PARSER,ROUTER,VERIFIER,EXPLAINER,MANIM agent
    class HITL_C,HITL_S hitl
    class TOOLS,RAG,WEB,CALC tool
    class APP,USER io
    class MCP,MANIM_SERVER server
```

---

## 📁 Project Structure

```
MathTutor/
├── src/
│   ├── backend/
│   │   ├── agents.py              # LangGraph graph, all agent nodes, HITL logic
│   │   ├── exceptions.py          # Agent_Exception with file + line info
│   │   ├── logger.py              # Structured logger
│   │   ├── tools/
│   │   │   └── tools.py           # rag_tool, web_search_tool, ingest_pdf, FAISS store
│   │   └── utils/
│   │       ├── artifacts.py       # Pydantic output schemas (Parser/Solver/Verifier/Explainer)
│   │       └── helper.py          # MediaProcessor (OCR/ASR), python_calculator sandbox
│   └── frontend/
│       └── new_app.py             # Streamlit UI, streaming, HITL widgets, activity panel
├── manim_mcp_server.py            # FastMCP server — renders Manim scenes to .mp4
├── .env                           # API keys (git-ignored)
├── .env.example                   # Template for .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔄 Agent Flow — Step by Step

1. **detect_input** — classifies input as `text`, `image`, or `audio`. Routes to OCR/ASR if needed.
2. **ocr_node / asr_node** — extracts text from image or audio using `MediaProcessor`.
3. **parser_agent** — cleans OCR/ASR noise, normalises math notation, identifies variables and constraints. If genuinely ambiguous → triggers `hitl_node`.
4. **hitl_node** *(clarification)* — `interrupt()` pauses the graph. User types clarification → resumed via `Command(resume=answer)`. Routes back to parser or continues to solver depending on context.
5. **intent_router** — classifies topic, difficulty, solver strategy. Decides if a Manim visualisation is needed.
6. **solver_agent ↔ tool_node** *(ReAct loop)* — LLaMA 3.3 70B with `bind_tools`. Each turn is either a tool call OR a final written answer — never mixed. Tools: `rag_tool` (PDF search), `web_search_tool`, `python_calculator_tool`.
7. **verifier_agent** — checks the solution for correctness, domain validity, edge cases. Routes to explainer (correct), retry solver (wrong), or HITL (uncertain).
8. **explainer_agent** — two separate LLM calls: ① structured `ExplainerOutput` (steps, key concepts, common mistakes) ② plain text Manim code (separate call to avoid Groq 400 on large code strings in function-calling schema).
9. **manim_node** — calls local FastMCP server asynchronously to render the Manim scene. Uses `nest_asyncio` to work inside Streamlit's event loop.
10. **satisfaction_hitl** — `interrupt()` asks if the student is satisfied. Yes → `END`. No → back to `parser_agent` with feedback.

---

## ⚙️ Setup

### 1. Clone & create environment

```bash
git clone https://github.com/your-username/MathTutor.git
cd MathTutor
python -m venv myenv
# Windows
myenv\Scripts\activate
# macOS / Linux
source myenv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key

# Optional — Manim MCP server
MANIM_MCP_SERVER_URL=http://localhost:8765/mcp
MANIM_OUTPUT_DIR=./manim_outputs
MANIM_SERVER_PORT=8765
```

### 4. (Optional) Start Manim MCP server

Required only if you want animated visualisations. Needs `manim` and `fastmcp` installed.

```bash
pip install manim fastmcp nest_asyncio
python manim_mcp_server.py
```

### 5. Run the app

```bash
streamlit run src/frontend/new_app.py
```

---

## 🔑 API Keys Required

| Service | Used for | Get it at |
|---|---|---|
| **Groq** | LLaMA 3.3 70B inference (all agents) | [console.groq.com](https://console.groq.com) |
| **Cohere** | PDF chunk embeddings (`embed-english-v3.0`) | [cohere.com](https://cohere.com) |

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | LLaMA 3.3 70B Versatile via Groq |
| **Agent orchestration** | LangGraph (`StateGraph`, `interrupt`, `InMemorySaver`) |
| **Frontend** | Streamlit |
| **Embeddings** | Cohere `embed-english-v3.0` (1024-dim) |
| **Vector search** | FAISS `IndexFlatIP` (cosine similarity) |
| **PDF ingestion** | LangChain `PyPDFLoader` + `RecursiveCharacterTextSplitter` |
| **Web search** | DuckDuckGo (`langchain_community`) |
| **Visualisation** | Manim Community Edition + FastMCP |
| **Async bridging** | `nest_asyncio` (Streamlit ↔ asyncio) |

---

## 🛡️ Known Limitations

- **In-memory RAG** — FAISS index is lost on Streamlit restart. Re-upload your PDF after restarting.
- **Groq rate limits** — `llama-3.3-70b-versatile` has token-per-minute limits. Heavy multi-tool solver turns may hit them.
- **Manim server** — must be running separately. If unavailable, the explanation still shows; only the video is skipped.
- **`np` not available in calculator** — the `python_calculator_tool` sandbox exposes `math` and `cmath` but not `numpy`. Use `math.sqrt` instead of `np.sqrt`.

---

## 📄 License

MIT License. See `LICENSE` for details.
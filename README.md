# CausalForge

**Executable science for discovering the next decisive experiment.**

CausalForge turns scattered science into executable causal worlds — helping researchers simulate interventions, expose contradictions, and discover the next decisive experiment.

> **[Read the Moonshot Paper](docs/moonshot-paper.md)** — the philosophical and technical foundation behind CausalForge, explaining why science needs an executable representation layer.

## Live Deployment

| Component | URL |
|-----------|-----|
| **Frontend** | https://causalforge.vercel.app |
| **Backend API** | https://causalforge-engine-production.up.railway.app |
| **Health Check** | https://causalforge-engine-production.up.railway.app/health |
| **GitHub** | https://github.com/Rohan5commit/causalforge |

## The Problem

Humanity stores science in documents. Documents are readable but not executable. This slows discovery and hides contradictions. We have extraordinary tools for finding papers (Google Scholar), reading papers (PubMed), and summarizing papers (AI chatbots). But we have built almost nothing for making science executable — for turning the relationships described in papers into structures we can simulate, test, and reason about.

## How CausalForge Works

### 1. Ingest Science
Feed research papers, abstracts, or structured notes. The system extracts structured claims, variables, mechanisms, and uncertainty language using NVIDIA NIM.

### 2. Build Causal World
Extracted claims are transformed into an explicit causal graph with:
- **Variables** (nodes) with types, descriptions, and uncertainty scores
- **Causal edges** with signs, strengths, confidence levels, and conflict indicators
- **Evidence links** back to source documents

### 3. Detect Contradictions
Automatically identifies:
- Direct conflicts between claims
- Incompatible mechanisms
- Evidence gaps
- Magnitude disputes

### 4. Simulate Interventions
Choose an intervention (e.g., "increase cool roof adoption to 60%") and observe predicted downstream effects with:
- Direction and magnitude of effect
- Confidence propagation through the causal chain
- Sensitivity to underlying uncertainty
- Evidence strength for each prediction

### 5. Discover Next Experiment
Information-gain scoring ranks experiments by:
- Expected reduction in model uncertainty
- Which contradictions they resolve
- What results would most change the field
- Estimated difficulty

## How Simulations Work

The simulator uses **deterministic BFS graph propagation** — not LLM inference. The math is performed entirely in Python:
1. Identify the intervention target variable
2. Apply the change (increase/decrease/binary) as a signed delta
3. BFS through outgoing causal edges, multiplying by sign, strength, and edge confidence
4. Discount confidence based on edge conflict levels
5. Return predicted effects with direction, magnitude, confidence, and full propagation traces

> NIM is only used optionally for natural-language explanation of results — the core simulation is **deterministic and inspectable**.

## How Experiment Ranking Works

Experiments are ranked by expected information gain:
- High-uncertainty edges get priority
- Active contradictions are weighted heavily
- Experiments that resolve multiple uncertainties rank higher
- Difficulty and feasibility are factored in

## Technology Stack

- **Frontend**: Next.js 15, React, TypeScript, Tailwind CSS v4, shadcn/ui
- **Backend**: Python FastAPI, Pydantic
- **AI Inference**: NVIDIA NIM (LLaMA 3.1 70B Instruct)
- **Data Store**: MongoDB Atlas
- **Visualization**: Custom SVG graph rendering with D3-inspired layout

## MongoDB Atlas

MongoDB Atlas stores and supports:
- Ingested document metadata
- Extracted scientific claims
- Graph node/edge state
- Contradiction records
- Simulation runs
- Experiment recommendations

Atlas provides the flexible document model ideal for heterogeneous scientific data, with the scalability to handle large corpora and the query power needed for evidence retrieval.

## Setup

### Prerequisites
- Node.js 18+
- Python 3.10+
- NVIDIA NIM API key
- MongoDB Atlas connection string

### Environment Variables

**Frontend** (`apps/web/.env.local`):
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

**Backend** (`services/causal-engine/.env`):
```
NIM_API_KEY=nvapi-your-key-here
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/causalforge
MONGODB_DB=causalforge
```

### Local Run

**Frontend:**
```bash
cd apps/web
npm install
npm run dev
# → http://localhost:3000
```

**Backend:**
```bash
cd services/causal-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
# → http://localhost:8000
```

## Demo Flow

1. Open http://localhost:3000
2. Click "Try Demo" on the landing page
3. Select the Urban Heat Mitigation domain
4. Click "Run Full Demo" to see the pipeline execute
5. Explore the Causal Map, Contradictions, Simulator, and Experiments pages
6. Use "Ask CausalForge" to query the model in natural language

## Limitations

- Demo mode uses preloaded data for instant experience
- Live extraction requires NVIDIA NIM API key
- Simulation uses deterministic BFS propagation through the causal graph (not full Bayesian inference)
- Graph layout is pre-computed for demo; production would use force-directed layout
- Evidence inspector shows demo excerpts; full version traces to actual paper sections

## Moonshot Paper

The full philosophical and technical foundation is documented in **[docs/moonshot-paper.md](docs/moonshot-paper.md)**. This paper explains the first-principles insight that *science is not a collection of facts but a web of causal relationships* — and why making that web executable is the key to faster discovery.

## Future Work

- Full Bayesian network inference for simulation
- Automated PDF parsing and entity extraction
- Real-time literature monitoring and model updates
- Multi-domain causal world models with cross-domain transfer
- Collaborative annotation and evidence voting
- Integration with experimental databases (ClinicalTrials.gov, arXiv)

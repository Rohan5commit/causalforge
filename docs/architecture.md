# CausalForge: System Architecture

## Overview

CausalForge is a monorepo with three major components:

```
CausalForge/
├── apps/web/              # Next.js 15 frontend
├── services/causal-engine/ # Python FastAPI backend
├── packages/shared/       # Shared TypeScript types
├── public/demo/           # Demo data assets
└── docs/                  # Documentation
```

## Component Architecture

### Frontend (Next.js 15)

```
apps/web/src/
├── app/
│   ├── layout.tsx          # Root layout with dark navigation
│   ├── page.tsx            # Landing page
│   ├── demo/page.tsx       # Demo pipeline visualization
│   ├── world-model/page.tsx # Causal graph interactive viewer
│   ├── contradictions/page.tsx # Contradiction inspector
│   ├── simulator/page.tsx  # Intervention simulator
│   ├── experiments/page.tsx # Next-best-experiment ranking
│   ├── ask/page.tsx        # Grounded Q&A assistant
│   └── thesis/page.tsx     # Moonshot thesis & architecture
├── components/ui/          # shadcn/ui components
└── lib/
    ├── api.ts              # API client for backend
    ├── demo-data.ts        # Preloaded demo domain data
    └── utils.ts            # Utility functions
```

### Backend (FastAPI)

```
services/causal-engine/
├── main.py                 # FastAPI application
├── .env                    # Environment variables
└── requirements.txt        # Python dependencies
```

**API Endpoints:**

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Health check |
| `/api/ingest` | POST | Ingest a scientific document |
| `/api/extract` | POST | Extract claims from a document |
| `/api/graph/build` | POST | Build causal graph from claims |
| `/api/contradictions/detect` | POST | Detect contradictions in claims |
| `/api/simulate` | POST | Run counterfactual simulation |
| `/api/experiments/rank` | POST | Rank next best experiments |
| `/api/chat` | POST | Grounded Q&A from world model |

### Shared Types

```
packages/shared/src/
└── index.ts                # All shared TypeScript types
```

## Data Flow

### Ingestion Flow
```
User Input → /api/ingest → MongoDB → Document ID returned
```

### Extraction Flow
```
Document → /api/extract → NVIDIA NIM → Structured Claims → MongoDB
```

### Graph Building Flow
```
Claims → /api/graph/build → NVIDIA NIM → Graph Validation → MongoDB
```

### Simulation Flow
```
Intervention + Graph → /api/simulate → Propagation Engine → Predicted Effects
```

### Experiment Ranking Flow
```
Graph + Contradictions + Simulations → /api/experiments/rank → Ranked Experiments
```

## External Services

### NVIDIA NIM
- **Base URL**: `https://integrate.api.nvidia.com/v1`
- **Model**: `nvidia/llama-3.1-70b-instruct`
- **Usage**: All AI inference (claim extraction, graph building, contradiction detection, simulation, experiment ranking, Q&A)
- **Format**: OpenAI-compatible API

### MongoDB Atlas
- **Driver**: Motor (async Python driver)
- **Collections**: documents, claims, graphs, contradictions, simulations, experiments
- **Usage**: Persistent storage for all extracted data, graph state, and results

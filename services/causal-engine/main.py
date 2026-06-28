"""
CausalForge – Causal Engine Backend
FastAPI service for scientific claim extraction, causal graph construction,
counterfactual simulation, contradiction detection, and experiment ranking.
"""

import os
import json
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
from motor.motor_asyncio import AsyncIOMotorClient
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────
NIM_API_KEY = os.getenv("NIM_API_KEY", "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")
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_MODEL = "nvidia/llama-3.1-70b-instruct"

MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb+srv://causalforge:causalforge@cluster0.abc123.mongodb.net/causalforge?retryWrites=true&w=majority"
)
MONGODB_DB = os.getenv("MONGODB_DB", "causalforge")

# ── Clients ─────────────────────────────────────────────────────────────
nim_client = OpenAI(base_url=NIM_BASE_URL, api_key=NIM_API_KEY)
mongo_client: Optional[AsyncIOMotorClient] = None
db = None

# ── FastAPI App ─────────────────────────────────────────────────────────
app = FastAPI(
    title="CausalForge Engine",
    version="0.1.0",
    description="AI-native scientific reasoning and causal world model engine",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    global mongo_client, db
    try:
        mongo_client = AsyncIOMotorClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        db = mongo_client[MONGODB_DB]
        await db.command("ping")
        print("✓ Connected to MongoDB Atlas")
    except Exception as e:
        print(f"⚠ MongoDB Atlas unavailable: {e}. Running without persistence.")
        mongo_client = None
        db = None


@app.on_event("shutdown")
async def shutdown():
    if mongo_client:
        mongo_client.close()


# ── Pydantic Models ─────────────────────────────────────────────────────

class DocumentIngestRequest(BaseModel):
    title: str
    abstract: str
    domain: str = "general"
    source: str = "paste"


class ClaimExtractionRequest(BaseModel):
    document_id: str
    title: str
    abstract: str
    domain: str = "general"


class GraphBuildRequest(BaseModel):
    claims: list[dict]
    domain: str = "general"


class SimulationRequest(BaseModel):
    intervention: dict
    world_model_id: str
    nodes: list[dict]
    edges: list[dict]


class ExperimentRankRequest(BaseModel):
    world_model_id: str
    nodes: list[dict]
    edges: list[dict]
    contradictions: list[dict]
    simulations: list[dict]


class ChatRequest(BaseModel):
    message: str
    world_model_id: str
    nodes: list[dict] = []
    edges: list[dict] = []
    claims: list[dict] = []
    contradictions: list[dict] = []


# ── NIM Helpers ──────────────────────────────────────────────────────────

def call_nim(messages: list[dict], temperature: float = 0.3, max_tokens: int = 4096, response_format: Optional[dict] = None) -> str:
    """Call NVIDIA NIM with retry on malformed output."""
    kwargs = {
        "model": NIM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.9,
        "max_tokens": max_tokens,
    }
    if response_format:
        kwargs["response_format"] = response_format

    for attempt in range(2):
        try:
            response = nim_client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            # Strip markdown fences if present
            if content.startswith("```"):
                lines = content.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                content = "\n".join(lines)
            return content.strip()
        except Exception as e:
            if attempt == 1:
                raise HTTPException(status_code=500, detail=f"NIM call failed: {e}")
            continue
    return ""


def extract_json_from_response(text: str) -> dict | list:
    """Parse JSON from NIM response, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in the text
        start = text.find("{")
        start_arr = text.find("[")
        if start >= 0 and (start_arr < 0 or start < start_arr):
            return json.loads(text[start:])
        elif start_arr >= 0:
            return json.loads(text[start_arr:])
        raise HTTPException(status_code=500, detail=f"Failed to parse JSON from NIM response: {text[:200]}")


# ── API Routes ──────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "causalforge-engine", "nim": "connected", "mongodb": "connected" if db else "unavailable"}


@app.post("/api/ingest")
async def ingest_document(req: DocumentIngestRequest):
    """Ingest a scientific document and return its ID."""
    doc_id = hashlib.md5(f"{req.title}{req.abstract}".encode()).hexdigest()[:12]
    doc = {
        "_id": doc_id,
        "title": req.title,
        "abstract": req.abstract,
        "domain": req.domain,
        "source": req.source,
        "ingestedAt": datetime.now(timezone.utc).isoformat(),
        "status": "processing",
    }
    if db:
        await db.documents.insert_one(doc)
    return {"document_id": doc_id, "status": "processing"}


@app.post("/api/extract")
async def extract_claims(req: ClaimExtractionRequest):
    """Extract scientific claims from a document using NVIDIA NIM."""
    extraction_prompt = f"""You are a scientific claim extraction system. Analyze the following scientific text and extract ALL structured claims.

Title: {req.title}
Domain: {req.domain}
Abstract: {req.abstract}

Return a JSON array of claims. Each claim must have these exact fields:
{{
  "id": "claim_<uuid>",
  "documentId": "{req.document_id}",
  "type": "causal|intervention|outcome|mechanism|correlation",
  "statement": "the extracted claim",
  "variables": ["list", "of", "variables"],
  "direction": "positive|negative|neutral|unclear",
  "confidence": 0.0-1.0,
  "evidenceStrength": "strong|moderate|weak|anecdotal",
  "uncertaintyLanguage": ["any", "hedging", "language"]
}}

Also return extracted variables as a separate list and any contradictions found.

Return JSON in this exact format:
{{
  "claims": [...],
  "variables": ["var1", "var2", ...],
  "contradictions": ["description of contradiction if found"]
}}"""

    response = call_nim([
        {"role": "system", "content": "You are a precise scientific claim extraction engine. Return only valid JSON."},
        {"role": "user", "content": extraction_prompt},
    ])

    result = extract_json_from_response(response)

    # Ensure valid structure
    if not isinstance(result, dict):
        result = {"claims": [], "variables": [], "contradictions": []}
    result.setdefault("claims", [])
    result.setdefault("variables", [])
    result.setdefault("contradictions", [])

    # Ensure all claims have required fields
    for i, claim in enumerate(result["claims"]):
        claim.setdefault("id", f"claim_{uuid.uuid4().hex[:8]}")
        claim.setdefault("documentId", req.document_id)
        claim.setdefault("type", "causal")
        claim.setdefault("direction", "unclear")
        claim.setdefault("confidence", 0.5)
        claim.setdefault("evidenceStrength", "moderate")
        claim.setdefault("uncertaintyLanguage", [])
        claim.setdefault("variables", [])

    if db:
        await db.documents.update_one(
            {"_id": req.document_id},
            {"$set": {"status": "extracted"}}
        )
        for claim in result["claims"]:
            claim["_id"] = claim["id"]
            await db.claims.insert_one(claim)

    return result


@app.post("/api/graph/build")
async def build_graph(req: GraphBuildRequest):
    """Build a causal graph from extracted claims."""
    claims_text = json.dumps(req.claims[:50], indent=2)  # limit for token efficiency

    build_prompt = f"""You are a causal graph construction engine. Given scientific claims, build an explicit causal graph.

Domain: {req.domain}
Claims: {claims_text}

Construct a causal graph with:

1. VARIABLES (nodes) - each with:
   - id: short unique id (e.g., "v_cool_roof")
   - label: human-readable name
   - type: "treatment"|"outcome"|"mediator"|"confounder"|"covariate"
   - description: brief description
   - uncertaintyScore: 0.0-1.0

2. EDGES (connections) - each with:
   - id: short unique id
   - source: variable id (cause)
   - target: variable id (effect)
   - sign: "positive"|"negative"|"unclear"
   - strength: 0.0-1.0
   - confidence: 0.0-1.0
   - conflictLevel: 0.0-1.0
   - mechanisms: list of mechanism descriptions
   - sourceClaimIds: list of claim ids supporting this edge

3. INTERVENTIONS - list of possible interventions:
   - id: short unique id
   - name: human-readable name
   - description: what this intervention does
   - targetVariable: which variable it targets
   - changeType: "increase"|"decrease"|"binary"
   - unit: optional unit of measurement

Return JSON:
{{
  "nodes": [...],
  "edges": [...],
  "interventions": [...]
}}"""

    response = call_nim([
        {"role": "system", "content": "You are a precise causal graph engine. Return only valid JSON with nodes, edges, and interventions."},
        {"role": "user", "content": build_prompt},
    ])

    result = extract_json_from_response(response)
    result.setdefault("nodes", [])
    result.setdefault("edges", [])
    result.setdefault("interventions", [])

    # Validate graph integrity deterministically
    node_ids = {n["id"] for n in result["nodes"]}
    valid_edges = []
    for edge in result["edges"]:
        if edge.get("source") in node_ids and edge.get("target") in node_ids:
            edge.setdefault("evidenceCount", 1)
            edge.setdefault("mechanisms", [])
            edge.setdefault("sourceClaimIds", [])
            valid_edges.append(edge)
    result["edges"] = valid_edges

    if db:
        graph_doc = {
            "_id": f"graph_{uuid.uuid4().hex[:12]}",
            "domain": req.domain,
            "nodes": result["nodes"],
            "edges": result["edges"],
            "interventions": result["interventions"],
            "createdAt": datetime.now(timezone.utc).isoformat(),
        }
        await db.graphs.insert_one(graph_doc)
        result["world_model_id"] = graph_doc["_id"]
    else:
        result["world_model_id"] = f"graph_{uuid.uuid4().hex[:12]}"

    return result


@app.post("/api/contradictions/detect")
async def detect_contradictions(req: GraphBuildRequest):
    """Detect contradictions in claims and graph structure."""
    claims_text = json.dumps(req.claims[:30], indent=2)

    detect_prompt = f"""You are a scientific contradiction detection engine. Analyze the claims and find ALL contradictions, conflicts, and evidence gaps.

Claims: {claims_text}

Find:
1. Direct conflicts: claims that directly contradict each other
2. Incompatible mechanisms: different mechanisms proposed for the same relationship
3. Evidence gaps: variables or relationships with insufficient evidence
4. Magnitude disputes: claims that disagree on the size/direction of effects

Return JSON:
{{
  "contradictions": [
    {{
      "id": "contra_<uuid>",
      "type": "direct_conflict|incompatible_mechanism|evidence_gap|magnitude_dispute",
      "description": "description of the contradiction",
      "involvedVariables": ["var1", "var2"],
      "claimA": {{"id": "claim_id", "statement": "...", "confidence": 0.7}},
      "claimB": {{"id": "claim_id", "statement": "...", "confidence": 0.6}},
      "severity": "critical|major|minor"
    }}
  ]
}}"""

    response = call_nim([
        {"role": "system", "content": "You are a precise contradiction detection engine. Return only valid JSON."},
        {"role": "user", "content": detect_prompt},
    ])

    result = extract_json_from_response(response)
    result.setdefault("contradictions", [])

    for c in result["contradictions"]:
        c.setdefault("id", f"contra_{uuid.uuid4().hex[:8]}")
        c.setdefault("resolved", False)
        c.setdefault("severity", "minor")

    if db:
        for c in result["contradictions"]:
            c["_id"] = c["id"]
            await db.contradictions.insert_one(c)

    return result


@app.post("/api/simulate")
async def simulate_intervention(req: SimulationRequest):
    """Run a counterfactual simulation for an intervention."""
    intervention = req.intervention
    nodes_text = json.dumps([n for n in req.nodes[:30]], indent=2)
    edges_text = json.dumps([e for e in req.edges[:50]], indent=2)

    sim_prompt = f"""You are a counterfactual simulation engine. Given a causal graph and an intervention, predict downstream effects.

Intervention: {json.dumps(intervention, indent=2)}

Causal Graph Nodes: {nodes_text}
Causal Graph Edges: {edges_text}

Simulate the intervention's effects by propagating through the causal graph.

For each affected variable, return:
- variableId: the variable being affected
- variableLabel: human-readable name
- direction: "positive"|"negative"|"no_change"|"uncertain"
- magnitude: 0.0-1.0 (normalized effect size)
- confidence: 0.0-1.0 (how confident in this prediction)
- causalPath: list of variable ids in the causal path
- evidenceStrength: "strong"|"moderate"|"weak"

Also return:
- overallConfidence: average confidence across all effects
- sensitivityToUncertainty: how much uncertainty in the graph affects results (0-1)

Return JSON:
{{
  "predictedEffects": [...],
  "overallConfidence": 0.0-1.0,
  "sensitivityToUncertainty": 0.0-1.0,
  "summary": "brief summary of simulation results"
}}"""

    response = call_nim([
        {"role": "system", "content": "You are a precise counterfactual simulation engine. Return only valid JSON with predicted effects."},
        {"role": "user", "content": sim_prompt},
    ])

    result = extract_json_from_response(response)
    result.setdefault("predictedEffects", [])
    result.setdefault("overallConfidence", 0.5)
    result.setdefault("sensitivityToUncertainty", 0.5)
    result.setdefault("summary", "")
    result["interventionId"] = intervention.get("id", "unknown")
    result["runAt"] = datetime.now(timezone.utc).isoformat()

    if db:
        sim_doc = {
            "_id": f"sim_{uuid.uuid4().hex[:12]}",
            "worldModelId": req.world_model_id,
            **result,
        }
        await db.simulations.insert_one(sim_doc)
        result["simulation_id"] = sim_doc["_id"]

    return result


@app.post("/api/experiments/rank")
async def rank_experiments(req: ExperimentRankRequest):
    """Rank the next best experiments to run."""
    nodes_text = json.dumps([n for n in req.nodes[:20]], indent=2)
    edges_text = json.dumps([e for e in req.edges[:30]], indent=2)
    contras_text = json.dumps([c for c in req.contradictions[:10]], indent=2)

    rank_prompt = f"""You are an experiment prioritization engine. Given a causal world model, identify the highest-value next experiments.

Causal Graph Nodes: {nodes_text}
Causal Graph Edges: {edges_text}
Known Contradictions: {contras_text}

Identify 3-5 experiments ranked by expected information gain. For each:

- rank: 1-5
- name: experiment name
- hypothesis: what we expect to learn
- variablesToMeasure: list of variables to measure
- expectedInformationGain: 0.0-1.0
- resolvesUncertaintyOf: list of variable or edge descriptions
- whyItMatters: explanation
- whatWouldChangeModel: what result would change our understanding
- estimatedDifficulty: "low"|"medium"|"high"

Return JSON:
{{
  "experiments": [...]
}}"""

    response = call_nim([
        {"role": "system", "content": "You are a precise experiment prioritization engine. Return only valid JSON."},
        {"role": "user", "content": rank_prompt},
    ])

    result = extract_json_from_response(response)
    result.setdefault("experiments", [])

    for i, exp in enumerate(result["experiments"]):
        exp.setdefault("id", f"exp_{uuid.uuid4().hex[:8]}")
        exp.setdefault("rank", i + 1)
        exp.setdefault("relatedContradictions", [])

    if db:
        for exp in result["experiments"]:
            exp["_id"] = exp["id"]
            exp["worldModelId"] = req.world_model_id
            await db.experiments.insert_one(exp)

    return result


@app.post("/api/chat")
async def chat_with_model(req: ChatRequest):
    """Answer questions grounded in the causal world model."""
    context_parts = []

    if req.nodes:
        context_parts.append(f"Variables: {json.dumps([{{'id': n.get('id'), 'label': n.get('label'), 'type': n.get('type'), 'uncertaintyScore': n.get('uncertaintyScore', 0)}} for n in req.nodes[:30]], indent=2)}")
    if req.edges:
        context_parts.append(f"Causal Relationships: {json.dumps([{{'source': e.get('source'), 'target': e.get('target'), 'sign': e.get('sign'), 'confidence': e.get('confidence'), 'conflictLevel': e.get('conflictLevel', 0)}} for e in req.edges[:30]], indent=2)}")
    if req.claims:
        context_parts.append(f"Scientific Claims: {json.dumps([{{'statement': c.get('statement'), 'confidence': c.get('confidence'), 'type': c.get('type')}} for c in req.claims[:20]], indent=2)}")
    if req.contradictions:
        context_parts.append(f"Known Contradictions: {json.dumps([{{'description': c.get('description'), 'severity': c.get('severity')}} for c in req.contradictions[:10]], indent=2)}")

    context = "\n\n".join(context_parts) if context_parts else "No world model loaded yet."

    chat_prompt = f"""You are CausalForge, a scientific reasoning assistant. You answer questions grounded ONLY in the causal world model, extracted claims, simulations, and evidence below. Do not invent information beyond what is provided.

CURRENT WORLD MODEL:
{context}

USER QUESTION: {req.message}

Answer precisely, reference specific variables and relationships from the model, and note where evidence is weak or contradictory."""

    response = call_nim([
        {"role": "system", "content": "You are CausalForge, a precise scientific reasoning assistant. Answer only from the provided world model context. Be direct and reference specific evidence."},
        {"role": "user", "content": chat_prompt},
    ], temperature=0.4)

    return {"response": response}


# ── Run ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

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
from pydantic import BaseModel
from openai import OpenAI
from motor.motor_asyncio import AsyncIOMotorClient

# ── Config ──────────────────────────────────────────────────────────────
NIM_API_KEY = os.getenv("NIM_API_KEY", "")
if not NIM_API_KEY:
    raise RuntimeError("NIM_API_KEY environment variable is required")
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_MODEL = "nvidia/llama-3.1-70b-instruct"

MONGODB_URI = os.getenv("MONGODB_URI", "")
if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI environment variable is required")
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
    allow_origins=["https://causalforge.vercel.app", "http://localhost:3000", "http://localhost:3099"],
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
    use_llm_explanation: bool = False  # opt-in NIM explanation after graph propagation


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


# ── Causal Graph Propagation Engine ───────────────────────────────────

def propagate_intervention(
    nodes: list[dict],
    edges: list[dict],
    intervention_var: str,
    delta: float = 1.0,
) -> dict:
    """Deterministic BFS propagation of an intervention through a causal graph.

    This is the core simulation algorithm. Given an intervention on a variable,
    it propagates effects forward through the directed causal graph, computing
    direction, magnitude, and confidence for each affected variable.

    Args:
        nodes: List of node dicts with 'id' and 'label' keys.
        edges: List of edge dicts with 'source', 'target', 'sign', 'strength',
                'confidence', and 'conflictLevel' keys.
        intervention_var: The variable ID being intervened on.
        delta: The magnitude of intervention (+1.0 for increase, -1.0 for decrease).

    Returns:
        dict with 'predictedEffects', 'overallConfidence', 'sensitivityToUncertainty',
        'propagationTraces', and 'summary'.
    """
    node_lookup = {n["id"]: n for n in nodes}
    # Build adjacency list: source -> [(edge, target)]
    adjacency: dict[str, list[tuple[dict, str]]] = {}
    for edge in edges:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        if src and tgt:
            adjacency.setdefault(src, []).append((edge, tgt))

    # BFS propagation using topological accumulation.
    # We collect ALL incoming effects per variable (multi-path convergence),
    # then propagate the aggregated effect forward.
    # Each queue entry: (variable_id, effect_value, cumulative_confidence, path, depth)
    queue = [(intervention_var, delta, 1.0, [intervention_var], 0)]
    # Track all incoming contributions per variable for convergence handling
    incoming_effects: dict[str, list[tuple[float, float, list[str]]]] = {intervention_var: [(delta, 1.0, [intervention_var])]} 
    processed = set()  # variables we've already propagated forward from

    while queue:
        var, effect, conf, path, depth = queue.pop(0)

        if var not in node_lookup:
            continue

        # For the intervention variable, always process.
        # For others, only process once we've collected all incoming contributions
        # (BFS level-order ensures parents are processed before children in a DAG).
        if var != intervention_var and var in processed:
            continue

        # Aggregate all incoming effects for this variable
        contributions = incoming_effects.get(var, [(effect, conf, path)])
        if var == intervention_var:
            agg_effect = effect
            agg_conf = conf
            best_path = path
        else:
            # Sum effects from all paths (convergent edges)
            agg_effect = sum(e for e, _, _ in contributions)
            # Confidence = max confidence across incoming paths (best evidence)
            agg_conf = max(c for _, c, _ in contributions)
            # Path = shortest path
            best_path = min(contributions, key=lambda x: len(x[2]))[2]

        processed.add(var)

        # Propagate to downstream variables
        for edge, target_id in adjacency.get(var, []):
            sign = 1.0 if edge.get("sign", "positive") == "positive" else -1.0
            strength = float(edge.get("strength", 0.5))
            edge_conf = float(edge.get("confidence", 0.5))
            conflict = float(edge.get("conflictLevel", 0.0))
            evidence_count = int(edge.get("evidenceCount", 1))

            # Reduce confidence based on conflict level
            adjusted_conf = edge_conf * (1.0 - conflict * 0.5)
            # Boost confidence slightly with more evidence (capped at 1.0)
            evidence_boost = min(evidence_count / 3.0, 1.0)
            adjusted_conf = adjusted_conf * (0.7 + 0.3 * evidence_boost)

            new_effect = agg_effect * sign * strength
            new_conf = agg_conf * adjusted_conf
            new_path = best_path + [target_id]

            # Record this incoming contribution for the target
            incoming_effects.setdefault(target_id, []).append((new_effect, new_conf, new_path))

            if depth + 1 <= 8:  # max propagation depth
                queue.append((target_id, new_effect, new_conf, new_path, depth + 1))

    # Build results from processed set (skip the intervention variable itself)
    predicted_effects = []
    confidences = []

    node_label = {n["id"]: n.get("label", n["id"]) for n in nodes}
    edge_uncertainty_scores = []
    final_effects: dict[str, tuple[float, float, list[str], int]] = {}

    # Aggregate final effects from all incoming contributions per variable
    for var_id, contributions in incoming_effects.items():
        if var_id == intervention_var:
            continue
        agg_effect = sum(e for e, _, _ in contributions)
        agg_conf = max(c for _, c, _ in contributions)
        best_path = min(contributions, key=lambda x: len(x[2]))[2]
        # Estimate depth from path length
        depth = len(best_path) - 1
        final_effects[var_id] = (agg_effect, agg_conf, best_path, depth)

    for var_id, (effect, conf, path, depth) in final_effects.items():
        if effect == 0.0:
            direction = "no_change"
        elif effect > 0:
            direction = "positive"
        else:
            direction = "negative"

        magnitude = min(abs(effect), 1.0)

        # Evidence strength based on confidence
        if conf >= 0.7:
            evidence_strength = "strong"
        elif conf >= 0.4:
            evidence_strength = "moderate"
        else:
            evidence_strength = "weak"

        predicted_effects.append({
            "variableId": var_id,
            "variableLabel": node_label.get(var_id, var_id),
            "direction": direction,
            "magnitude": round(magnitude, 4),
            "confidence": round(conf, 4),
            "causalPath": path,
            "pathLength": len(path) - 1,
            "evidenceStrength": evidence_strength,
        })
        confidences.append(conf)

    # Compute overall metrics
    overall_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

    # Sensitivity to uncertainty: avg of conflict levels on traversal edges
    for edge in edges:
        edge_uncertainty_scores.append(float(edge.get("conflictLevel", 0.0)) +
                                       float(1.0 - edge.get("confidence", 0.5)))
    sensitivity = round(
        sum(edge_uncertainty_scores) / len(edge_uncertainty_scores), 4
    ) if edge_uncertainty_scores else 0.5

    # Build propagation traces for debugging
    traces = []
    for var_id, (effect, conf, path, depth) in final_effects.items():
        traces.append({
            "variable": var_id,
            "label": node_label.get(var_id, var_id),
            "effect": round(effect, 4),
            "confidence": round(conf, 4),
            "path": " → ".join(path),
            "depth": depth,
        })
    traces.sort(key=lambda t: t["depth"])

    # Generate deterministic summary
    if not predicted_effects:
        summary = (f"No downstream effects found for intervention on '{intervention_var}'. "
                   "The variable may be a leaf node with no outgoing causal edges.")
    else:
        pos = [e for e in predicted_effects if e["direction"] == "positive"]
        neg = [e for e in predicted_effects if e["direction"] == "negative"]
        parts = []
        if pos:
            parts.append(f"{len(pos)} positive effect(s): {', '.join(e['variableLabel'] for e in pos[:3])}")
        if neg:
            parts.append(f"{len(neg)} negative effect(s): {', '.join(e['variableLabel'] for e in neg[:3])}")
        summary = (f"Intervention on '{intervention_var}' propagates to {len(predicted_effects)} "
                   f"variable(s). {' '.join(parts)} "
                   f"Overall confidence: {overall_confidence:.0%}.")

    return {
        "predictedEffects": predicted_effects,
        "overallConfidence": overall_confidence,
        "sensitivityToUncertainty": sensitivity,
        "propagationTraces": traces,
        "summary": summary,
    }


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
    """Run a counterfactual simulation via deterministic BFS graph propagation.

    The core math (BFS through edges, signed/strength-weighted propagation)
    is performed deterministically in Python. NIM is only used optionally
    for natural-language explanation of results.
    """
    intervention = req.intervention
    target_var = intervention.get("targetVariable") or intervention.get("target_variable") or intervention.get("id", "")
    change_type = intervention.get("changeType", "increase")
    delta = -1.0 if change_type == "decrease" else 1.0

    # ── 1. Deterministic graph propagation (pure Python, no LLM) ──────
    result = propagate_intervention(
        nodes=req.nodes,
        edges=req.edges,
        intervention_var=target_var,
        delta=delta,
    )

    result["interventionId"] = intervention.get("id", "unknown")
    result["interventionTarget"] = target_var
    result["runAt"] = datetime.now(timezone.utc).isoformat()
    result["method"] = "deterministic_bfs_propagation"  # mark as graph-based, not LLM

    # ── 2. Optional: NIM-generated natural-language explanation ────────
    if req.use_llm_explanation:
        effects_text = json.dumps(result["predictedEffects"], indent=2)
        traces_text = json.dumps(result["propagationTraces"][:10], indent=2)
        explanation_prompt = f"""You are a causal reasoning assistant. Given the deterministic simulation results below,
provide a brief, precise natural-language explanation of what this intervention does.

Intervention: {json.dumps(intervention, indent=2)}
Predicted Effects: {effects_text}
Propagation Traces: {traces_text}

Write a 2-3 sentence explanation of the causal chain and key findings."""
        try:
            explanation = call_nim([
                {"role": "system", "content": "You are a precise causal reasoning assistant. Be concise."},
                {"role": "user", "content": explanation_prompt},
            ], temperature=0.2, max_tokens=512)
            result["llmExplanation"] = explanation
        except Exception:
            pass  # explanation is optional

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
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

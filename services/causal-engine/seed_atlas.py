"""
CausalForge – MongoDB Atlas Seed Script
Seeds the Urban Heat Mitigation demo domain into MongoDB Atlas.
Run: cd services/causal-engine && source .venv/bin/activate && python seed_atlas.py
"""

import asyncio
import os
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "causalforge")

# ── Demo Data ──────────────────────────────────────────────────────────

DOCUMENTS = [
    {
        "_id": "doc_cool_roof_01",
        "title": "Cool Roof Adoption and Urban Temperature Reduction: A Meta-Analysis",
        "authors": ["Chen et al.", "Urban Climate Journal"],
        "abstract": "Cool roofs with high solar reflectance (albedo > 0.65) reduce surface temperatures by 10-15°C and ambient air temperatures by 0.5-2°C in dense urban environments. The effect scales with adoption rate, with diminishing returns above 60% coverage in tropical climates. Energy savings from reduced cooling loads range from 10-30% depending on building type and climate zone.",
        "domain": "urban-heat",
        "source": "demo",
        "ingestedAt": datetime.now(timezone.utc).isoformat(),
        "status": "extracted",
    },
    {
        "_id": "doc_tree_canopy_02",
        "title": "Urban Tree Canopy and Street-Level Thermal Comfort: Mechanisms and Limits",
        "authors": ["Park & Williams", "Environmental Research Letters"],
        "abstract": "Urban tree canopy provides cooling through evapotranspiration and shading, reducing local air temperatures by 1-5°C and radiant heat exposure by 20-40%. However, increased tree cover in poorly ventilated urban canyons can trap humidity, potentially reducing thermal comfort in humid subtropical climates.",
        "domain": "urban-heat",
        "source": "demo",
        "ingestedAt": datetime.now(timezone.utc).isoformat(),
        "status": "extracted",
    },
    {
        "_id": "doc_urban_albedo_03",
        "title": "Surface Albedo Modification and City-Scale Energy Balance",
        "authors": ["Santamouris et al.", "Nature Energy"],
        "abstract": "Increasing urban surface albedo through reflective pavements and materials reduces the urban heat island intensity by 0.3-1.2°C city-wide. The relationship is nonlinear: initial albedo increases show larger temperature reductions, with saturation effects beyond albedo 0.5.",
        "domain": "urban-heat",
        "source": "demo",
        "ingestedAt": datetime.now(timezone.utc).isoformat(),
        "status": "extracted",
    },
    {
        "_id": "doc_energy_feedback_04",
        "title": "Urban Heat, Energy Consumption, and the Feedback Loop in Mega-Cities",
        "authors": ["Li & Zhang", "Applied Energy"],
        "abstract": "Urban heat increases energy consumption for cooling by 1.5-3% per degree Celsius of temperature rise, creating a feedback loop where air conditioning exhaust further heats the outdoor environment. This positive feedback can amplify initial temperature increases by 10-25% in dense commercial districts.",
        "domain": "urban-heat",
        "source": "demo",
        "ingestedAt": datetime.now(timezone.utc).isoformat(),
        "status": "extracted",
    },
]

CLAIMS = [
    {"_id": "claim_cool_roof_temp", "documentId": "doc_cool_roof_01", "domain": "urban-heat", "type": "causal", "statement": "Cool roofs reduce surface temperatures by 10-15°C and ambient air temperatures by 0.5-2°C", "variables": ["v_cool_roof", "v_surface_albedo", "v_uhi_intensity"], "direction": "negative", "confidence": 0.9, "evidenceStrength": "strong", "uncertaintyLanguage": []},
    {"_id": "claim_cool_roof_energy", "documentId": "doc_cool_roof_01", "domain": "urban-heat", "type": "outcome", "statement": "Energy savings from reduced cooling loads range from 10-30% depending on building type", "variables": ["v_cool_roof", "v_energy_cooling"], "direction": "negative", "confidence": 0.85, "evidenceStrength": "strong", "uncertaintyLanguage": ["depending on"]},
    {"_id": "claim_cool_roof_diminish", "documentId": "doc_cool_roof_01", "domain": "urban-heat", "type": "mechanism", "statement": "Cool roof effectiveness shows diminishing returns above 60% coverage in tropical climates", "variables": ["v_cool_roof", "v_surface_albedo"], "direction": "positive", "confidence": 0.75, "evidenceStrength": "moderate", "uncertaintyLanguage": ["diminishing"]},
    {"_id": "claim_tree_cool", "documentId": "doc_tree_canopy_02", "domain": "urban-heat", "type": "causal", "statement": "Urban tree canopy reduces local air temperatures by 1-5°C and radiant heat exposure by 20-40%", "variables": ["v_tree_canopy", "v_uhi_intensity", "v_thermal_comfort"], "direction": "negative", "confidence": 0.8, "evidenceStrength": "strong", "uncertaintyLanguage": []},
    {"_id": "claim_tree_humid", "documentId": "doc_tree_canopy_02", "domain": "urban-heat", "type": "mechanism", "statement": "Increased tree cover in poorly ventilated urban canyons can trap humidity, reducing thermal comfort", "variables": ["v_tree_canopy", "v_humidity", "v_thermal_comfort"], "direction": "negative", "confidence": 0.7, "evidenceStrength": "moderate", "uncertaintyLanguage": ["can", "potentially"]},
    {"_id": "claim_albedo_uhi", "documentId": "doc_urban_albedo_03", "domain": "urban-heat", "type": "causal", "statement": "Increasing urban surface albedo reduces UHI intensity by 0.3-1.2°C city-wide", "variables": ["v_surface_albedo", "v_uhi_intensity"], "direction": "negative", "confidence": 0.85, "evidenceStrength": "strong", "uncertaintyLanguage": []},
    {"_id": "claim_albedo_combined", "documentId": "doc_urban_albedo_03", "domain": "urban-heat", "type": "intervention", "statement": "Combined albedo and vegetation strategies achieve 2-4°C reduction in peak temperatures", "variables": ["v_surface_albedo", "v_tree_canopy", "v_uhi_intensity"], "direction": "negative", "confidence": 0.8, "evidenceStrength": "moderate", "uncertaintyLanguage": []},
    {"_id": "claim_ac_feedback", "documentId": "doc_energy_feedback_04", "domain": "urban-heat", "type": "mechanism", "statement": "Urban heat increases cooling energy consumption by 1.5-3% per degree Celsius, creating a feedback loop", "variables": ["v_uhi_intensity", "v_energy_cooling"], "direction": "positive", "confidence": 0.75, "evidenceStrength": "moderate", "uncertaintyLanguage": ["can amplify"]},
    {"_id": "claim_ac_amplify", "documentId": "doc_energy_feedback_04", "domain": "urban-heat", "type": "outcome", "statement": "AC exhaust can amplify initial temperature increases by 10-25% in dense commercial districts", "variables": ["v_energy_cooling", "v_uhi_intensity"], "direction": "positive", "confidence": 0.6, "evidenceStrength": "weak", "uncertaintyLanguage": ["can"]},
]

VARIABLES = [
    {"_id": "v_cool_roof", "label": "Cool Roof Adoption Rate", "type": "treatment", "description": "Percentage of rooftops with high-albedo reflective materials", "uncertaintyScore": 0.15, "domain": "urban-heat"},
    {"_id": "v_tree_canopy", "label": "Urban Tree Canopy Coverage", "type": "treatment", "description": "Percentage of urban area covered by tree canopy", "uncertaintyScore": 0.2, "domain": "urban-heat"},
    {"_id": "v_surface_albedo", "label": "Urban Surface Albedo", "type": "mediator", "description": "Average reflectivity of all urban surfaces", "uncertaintyScore": 0.25, "domain": "urban-heat"},
    {"_id": "v_uhi_intensity", "label": "Urban Heat Island Intensity", "type": "outcome", "description": "Temperature difference between urban and rural baseline", "uncertaintyScore": 0.3, "domain": "urban-heat"},
    {"_id": "v_air_temp", "label": "Ambient Air Temperature", "type": "outcome", "description": "Street-level ambient air temperature", "uncertaintyScore": 0.1, "domain": "urban-heat"},
    {"_id": "v_energy_cooling", "label": "Cooling Energy Demand", "type": "outcome", "description": "Energy consumed for air conditioning", "uncertaintyScore": 0.15, "domain": "urban-heat"},
    {"_id": "v_thermal_comfort", "label": "Pedestrian Thermal Comfort", "type": "outcome", "description": "Outdoor thermal comfort index for pedestrians", "uncertaintyScore": 0.35, "domain": "urban-heat"},
    {"_id": "v_humidity", "label": "Urban Humidity Level", "type": "mediator", "description": "Relative humidity in urban canyon environments", "uncertaintyScore": 0.4, "domain": "urban-heat"},
]

EDGES = [
    {"_id": "e1", "source": "v_cool_roof", "target": "v_surface_albedo", "sign": "positive", "strength": 0.85, "confidence": 0.9, "conflictLevel": 0.05, "mechanisms": ["Increases roof reflectivity"], "sourceClaimIds": ["claim_cool_roof_temp"], "evidenceCount": 12},
    {"_id": "e2", "source": "v_surface_albedo", "target": "v_uhi_intensity", "sign": "negative", "strength": 0.7, "confidence": 0.85, "conflictLevel": 0.1, "mechanisms": ["Reduces absorbed solar radiation"], "sourceClaimIds": ["claim_albedo_uhi"], "evidenceCount": 15},
    {"_id": "e3", "source": "v_tree_canopy", "target": "v_uhi_intensity", "sign": "negative", "strength": 0.6, "confidence": 0.8, "conflictLevel": 0.15, "mechanisms": ["Evapotranspiration cooling", "Shading reduces surface heating"], "sourceClaimIds": ["claim_tree_cool"], "evidenceCount": 18},
    {"_id": "e4", "source": "v_tree_canopy", "target": "v_humidity", "sign": "positive", "strength": 0.5, "confidence": 0.7, "conflictLevel": 0.3, "mechanisms": ["Transpiration adds moisture"], "sourceClaimIds": ["claim_tree_humid"], "evidenceCount": 8},
    {"_id": "e5", "source": "v_humidity", "target": "v_thermal_comfort", "sign": "negative", "strength": 0.45, "confidence": 0.65, "conflictLevel": 0.35, "mechanisms": ["High humidity reduces evaporative cooling from skin"], "sourceClaimIds": ["claim_tree_humid"], "evidenceCount": 6},
    {"_id": "e6", "source": "v_uhi_intensity", "target": "v_air_temp", "sign": "positive", "strength": 0.95, "confidence": 0.95, "conflictLevel": 0.02, "mechanisms": ["Direct temperature increase"], "sourceClaimIds": ["claim_cool_roof_temp"], "evidenceCount": 25},
    {"_id": "e7", "source": "v_air_temp", "target": "v_energy_cooling", "sign": "positive", "strength": 0.8, "confidence": 0.9, "conflictLevel": 0.08, "mechanisms": ["Higher temps increase cooling loads"], "sourceClaimIds": ["claim_ac_feedback"], "evidenceCount": 20},
    {"_id": "e8", "source": "v_energy_cooling", "target": "v_uhi_intensity", "sign": "positive", "strength": 0.3, "confidence": 0.6, "conflictLevel": 0.4, "mechanisms": ["AC exhaust heat release"], "sourceClaimIds": ["claim_ac_amplify"], "evidenceCount": 5},
    {"_id": "e9", "source": "v_cool_roof", "target": "v_energy_cooling", "sign": "negative", "strength": 0.65, "confidence": 0.85, "conflictLevel": 0.1, "mechanisms": ["Reduces indoor heat gain"], "sourceClaimIds": ["claim_cool_roof_energy"], "evidenceCount": 14},
    {"_id": "e10", "source": "v_tree_canopy", "target": "v_thermal_comfort", "sign": "positive", "strength": 0.55, "confidence": 0.75, "conflictLevel": 0.2, "mechanisms": ["Shading provides direct comfort benefit"], "sourceClaimIds": ["claim_tree_cool"], "evidenceCount": 10},
]

CONTRADICTIONS = [
    {"_id": "contra_humidity_comfort", "type": "incompatible_mechanism", "description": "Tree canopy cools through evapotranspiration but simultaneously increases humidity, which can reduce thermal comfort in humid climates.", "involvedVariables": ["v_tree_canopy", "v_humidity", "v_thermal_comfort"], "claimA": {"id": "claim_tree_cool", "statement": "Tree canopy reduces ambient temperature by 1-5°C through evapotranspiration", "confidence": 0.85}, "claimB": {"id": "claim_tree_humid", "statement": "Increased tree cover in poorly ventilated urban canyons can trap humidity and reduce thermal comfort", "confidence": 0.7}, "severity": "major", "resolved": False},
    {"_id": "contra_energy_feedback", "type": "magnitude_dispute", "description": "Air conditioning exhaust creates a positive feedback loop, but estimates of its magnitude vary from 10-25% amplification.", "involvedVariables": ["v_energy_cooling", "v_uhi_intensity"], "claimA": {"id": "claim_ac_feedback", "statement": "AC exhaust amplifies initial temperature increases by 10-25%", "confidence": 0.6}, "claimB": {"id": "claim_ac_passive", "statement": "Passive cooling strategies can break the energy-heat feedback loop entirely", "confidence": 0.55}, "severity": "minor", "resolved": False},
    {"_id": "contra_albedo_saturation", "type": "magnitude_dispute", "description": "Cool roof effectiveness shows diminishing returns above 60% adoption in tropical climates.", "involvedVariables": ["v_cool_roof", "v_surface_albedo"], "claimA": {"id": "claim_albedo_diminish", "statement": "Cool roof effectiveness shows diminishing returns above 60% adoption", "confidence": 0.75}, "claimB": {"id": "claim_albedo_linear", "statement": "Reflective materials provide linear temperature reduction up to 80% coverage", "confidence": 0.7}, "severity": "minor", "resolved": False},
]

GRAPH = {
    "_id": "graph_urban_heat_demo",
    "domain": "urban-heat",
    "nodes": VARIABLES,
    "edges": EDGES,
    "interventions": [
        {"_id": "int_cool_roof_60", "name": "Increase Cool Roof Adoption to 60%", "description": "Mandate cool roof materials for 60% of urban rooftop area", "targetVariable": "v_cool_roof", "changeType": "increase", "unit": "%"},
        {"_id": "int_tree_canopy_35", "name": "Expand Tree Canopy to 35%", "description": "Plant trees and protect existing canopy to achieve 35% coverage", "targetVariable": "v_tree_canopy", "changeType": "increase", "unit": "%"},
        {"_id": "int_combined", "name": "Combined Cool Roofs + Trees", "description": "Simultaneously deploy cool roofs (50%) and expand tree canopy (30%)", "targetVariable": "v_surface_albedo", "changeType": "increase", "unit": "composite"},
    ],
    "createdAt": datetime.now(timezone.utc).isoformat(),
}


async def seed():
    if not MONGODB_URI:
        print("⚠ MONGODB_URI not set. Cannot seed Atlas.")
        return

    print(f"Connecting to MongoDB Atlas...")
    client = AsyncIOMotorClient(MONGODB_URI, serverSelectionTimeoutMS=10000)
    db = client[MONGODB_DB]

    try:
        await db.command("ping")
        print("✓ Connected to MongoDB Atlas")
    except Exception as e:
        print(f"✗ Cannot connect to Atlas: {e}")
        return

    collections = {
        "documents": DOCUMENTS,
        "claims": CLAIMS,
        "variables": VARIABLES,
        "edges": EDGES,
        "contradictions": CONTRADICTIONS,
        "graphs": [GRAPH],
    }

    for name, data in collections.items():
        col = db[name]
        await col.delete_many({})
        if data:
            await col.insert_many(data)
        print(f"  ✓ Seeded {len(data)} records into '{name}'")

    # Create standard indexes
    await db.documents.create_index("domain")
    await db.claims.create_index("documentId")
    await db.claims.create_index("type")
    await db.edges.create_index("source")
    await db.edges.create_index("target")
    await db.contradictions.create_index("severity")
    await db.graphs.create_index("domain")
    print("  ✓ Created standard indexes")

    # Create Atlas Search indexes
    # 1. Vector Search index for semantic similarity on claim embeddings
    try:
        await db.claims.create_search_index({
            "name": "claim_vector_index",
            "definition": {
                "fields": [{
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 1024,
                    "similarity": "cosine",
                }]
            },
        })
        print("  ✓ Created claim_vector_index (Atlas Vector Search)")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("  ✓ claim_vector_index already exists")
        else:
            print(f"  ⚠ Vector index: {e}")

    # 2. Full-text Search index for BM25 claim retrieval
    try:
        await db.claims.create_search_index({
            "name": "claim_text_index",
            "definition": {
                "mappings": {
                    "dynamic": False,
                    "fields": {
                        "statement": {"type": "string", "analyzer": "luceneStandard"},
                        "type": {"type": "string", "analyzer": "luceneKeyword"},
                    },
                }
            },
        })
        print("  ✓ Created claim_text_index (Atlas Search)")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("  ✓ claim_text_index already exists")
        else:
            print(f"  ⚠ Text index: {e}")

    # Generate embeddings for claims using NIM embedding model
    print("  Generating claim embeddings...")
    try:
        from openai import OpenAI
        nim_key = os.getenv("NIM_API_KEY", "")
        if nim_key:
            nim = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=nim_key)
            statements = [c["statement"] for c in CLAIMS]
            resp = nim.embeddings.create(model="nvidia/nv-embedqa-e5-v5", input=statements, encoding_format="float")
            for claim, embedding in zip(CLAIMS, resp.data):
                await db.claims.update_one(
                    {"_id": claim["_id"]},
                    {"$set": {"embedding": embedding.embedding}},
                )
            print(f"  ✓ Generated embeddings for {len(CLAIMS)} claims (1024-dim, nv-embedqa-e5-v5)")
        else:
            print("  ⚠ NIM_API_KEY not set — skipping embedding generation")
    except Exception as e:
        print(f"  ⚠ Embedding generation failed: {e}")

    # Verify
    for name in collections:
        count = await db[name].count_documents({})
        print(f"  ✓ Verified: {name} has {count} documents")

    # Check embedding coverage
    with_embed = await db.claims.count_documents({"embedding": {"$exists": True}})
    total_claims = await db.claims.count_documents({})
    print(f"  ✓ Claims with embeddings: {with_embed}/{total_claims}")

    print("\n✓ Atlas seed complete. Database: causalforge")
    print("  Features: Vector Search (semantic), Atlas Search (full-text), Claim Embeddings")
    client.close()


if __name__ == "__main__":
    asyncio.run(seed())

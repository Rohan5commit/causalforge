# CausalForge: Devpost Copy

## Project Summary
CausalForge is an AI-native scientific reasoning system that turns fragmented research papers into executable causal world models. It lets researchers inspect causal structure, detect contradictions across literature, run counterfactual intervention simulations, and identify the highest-information-gain experiment to run next.

## The Problem
Humanity stores science in documents. Documents are readable but not executable. This fundamental limitation means:
- Contradictions between papers go undetected for years
- Experiment design relies on intuition rather than computation
- The same dead-end experiments are repeated across labs worldwide
- Cross-domain causal patterns remain invisible
- Researchers waste time synthesizing knowledge that could be computed

## The Solution
CausalForge builds an explicit causal world model from scientific literature:
1. **Ingest** papers and extract structured claims using NVIDIA NIM
2. **Build** a causal graph with variables, edges, signs, and uncertainty scores
3. **Detect** contradictions, conflicts, and evidence gaps automatically
4. **Simulate** counterfactual interventions with confidence propagation
5. **Rank** the next best experiment by expected information gain

## Why This Is Zero-to-One
This is not a search engine, literature review tool, or chatbot. It is a new category: **executable science infrastructure**. No existing product converts scientific literature into simulation-capable causal models with contradiction detection and experiment ranking.

## Why It Matters
If CausalForge succeeds:
- **Faster science**: Contradictions detected within hours of publication
- **Less waste**: Experiments selected by information gain, not intuition
- **Better decisions**: Interventions evaluated against causal models
- **Cross-domain discovery**: Causal patterns that transfer between fields become visible

## Tech Stack
- Next.js 15, React, TypeScript (frontend)
- Python FastAPI, Pydantic (backend)
- NVIDIA NIM LLaMA 3.1 70B (AI inference)
- MongoDB Atlas (data storage)
- Custom SVG graph visualization

## What We Built
- Full 8-page application with dark mode UI
- Working demo with preloaded urban heat mitigation domain
- Interactive causal graph with node inspection
- Contradiction detector with severity classification
- Counterfactual simulator with confidence propagation
- Next-best-experiment engine with information-gain scoring
- Grounded Q&A assistant that answers only from the world model
- Complete documentation including Moonshot Paper

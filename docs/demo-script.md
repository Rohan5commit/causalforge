# CausalForge: Demo Script (2-3 minutes)

## Setup
- Screen shows CausalForge landing page at causalforge.dev
- Dark mode, clean typography, gradient accents

## 0:00-0:20 — Opening
"Humanity stores science in documents. But documents are readable, not executable. You can't simulate a paper. You can't intervene on a paper. CausalForge changes that."

[Click "Try Demo"]

## 0:20-0:40 — Domain Selection
"We start with a curated domain: Urban Heat Mitigation. Four research papers, preloaded and ready."

[Show domain card with 4 papers listed]

"Each paper has been analyzed and its scientific claims extracted."

[Click "Run Full Demo"]

## 0:40-1:10 — Pipeline Animation
"Watch the pipeline: ingest documents, extract scientific claims with NVIDIA NIM, build the explicit causal graph, detect contradictions, run simulations."

[Pipeline animation plays — three steps with checkmarks]

"The system extracted 11 claims, 8 variables, 10 causal edges, and found 3 contradictions."

## 1:10-1:40 — Causal Graph
[Click "Explore Causal Graph"]

"Here's the causal world model. Every node is a variable — cool roofs, tree canopy, surface albedo, heat island intensity. Every edge is a causal relationship with a sign, strength, and confidence score."

[Hover over an edge]

"See this edge? Cool roof adoption to surface albedo — 90% confidence, strong evidence, low conflict. But look at this one — tree canopy to humidity to thermal comfort — 35% conflict. The literature disagrees here."

[Click a node]

"Click any variable to see its connected relationships, evidence sources, and uncertainty level."

## 1:40-2:00 — Contradictions
[Click "View Contradictions"]

"The system found 3 contradictions. The biggest: tree canopy cools through evapotranspiration but simultaneously increases humidity, which can reduce thermal comfort in humid cities. This is a major unresolved conflict."

## 2:00-2:20 — Simulation
[Click "Run Simulation"]

"Now let's simulate an intervention. We'll increase cool roof adoption to 60%."

[Run simulation]

"Predicted effects: surface albedo increases by 75%, heat island intensity drops, cooling energy demand falls. 83% overall confidence. But thermal comfort improvement is only moderate — because the humidity path introduces uncertainty."

## 2:20-2:40 — Next Experiment
[Click "Next Best Experiment"]

"The system recommends measuring the humidity-comfort tradeoff in urban canyons as the #1 next experiment. 85% expected information gain. It resolves the largest active contradiction and could redirect urban planning for tropical cities worldwide."

## 2:40-3:00 — Close
"This is CausalForge. Not a chatbot. Not a search engine. A system for making science executable. The future of research is not just AI answering questions about papers — it's AI turning science into manipulable causal worlds."

[Show landing page with "Enter the Causal World" CTA]

## Key Talking Points
1. **This is new infrastructure**, not another tool for reading papers
2. **Explicit causal graphs** — not hidden in a prompt, fully inspectable
3. **Contradiction detection** — finds what humans miss in thousands of papers
4. **Simulation** — not narrative, actual propagation through causal structure
5. **Experiment ranking** — information-gain scoring, not intuition

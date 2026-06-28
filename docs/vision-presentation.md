# CausalForge: Vision Presentation

## Slide 1: Opening

**Humanity stores science in documents.**
**Documents are readable but not executable.**

Every research paper describes causal relationships between variables. But you cannot simulate a paper. You cannot intervene on a paper. You cannot compute what a paper would predict under conditions it never considered.

## Slide 2: The Problem

- 3 million papers published annually
- Contradictions go undetected for years
- Experiment design relies on intuition, not computation
- The same dead-end experiments are repeated across labs worldwide
- Cross-domain causal patterns remain invisible

**We have tools for finding papers, reading papers, and summarizing papers. We have no tools for making science executable.**

## Slide 3: Why Now

- AI can extract structured claims from unstructured text
- Causal inference theory provides the mathematical foundation
- Graph databases can store and query causal structures
- GPU-accelerated inference enables real-time simulation
- The scientific community is desperate for better tools

## Slide 4: First-Principles Insight

**Science is not a collection of facts. It is a web of causal relationships.**

If you represent that web explicitly, you can:
- Simulate what happens when you intervene
- Detect where scientists disagree
- Compute which experiment would change the field most

## Slide 5: The System

**CausalForge Architecture:**

1. **Ingest** → Papers, abstracts, structured notes
2. **Extract** → Variables, mechanisms, causal claims (NVIDIA NIM)
3. **Build** → Explicit causal graph with uncertainty scores
4. **Detect** → Contradictions, conflicts, evidence gaps
5. **Simulate** → Counterfactual interventions with confidence propagation
6. **Rank** → Next best experiment by information gain

## Slide 6: Demo Walkthrough

**Domain: Urban Heat Mitigation**

- 4 research papers ingested
- 11 scientific claims extracted
- 8 variables identified
- 10 causal edges constructed
- 3 contradictions detected
- 3 interventions simulated
- 3 experiments ranked

**Key finding: Measuring the humidity-comfort tradeoff in urban canyons is the #1 next experiment — it resolves the largest active contradiction and could redirect urban planning for tropical cities worldwide.**

## Slide 7: Technical Depth

- **Explicit causal graph**: Not hidden in a prompt. Every variable, edge, and sign is inspectable.
- **Evidence traces**: Every relationship traces back to source documents and excerpts.
- **Uncertainty scoring**: Every node and edge carries a confidence and conflict level.
- **Simulation engine**: Propagation through causal structure, not narrative generation.
- **Information-gain ranking**: Mathematical prioritization of experiments.

## Slide 8: Long-Term Future

In 10 years, CausalForge could become:

- **The operating system for scientific research**
- Every paper automatically ingested into global causal models
- Contradictions detected within hours of publication
- Drug discovery accelerated by simulation before experimentation
- Climate interventions evaluated against causal world models
- Cross-domain causal patterns discovered by AI

**Faster science. Less wasted experimentation. Better intervention discovery.**

## Slide 9: Close

CausalForge is a first prototype of executable science.

The question is not whether science will become executable.
The question is who builds the first real system.

**CausalForge.**

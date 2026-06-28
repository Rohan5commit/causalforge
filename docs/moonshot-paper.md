# CausalForge: Executable Science for Discovering the Next Decisive Experiment

## Abstract

Humanity stores science in documents. Documents are readable but not executable. This fundamental limitation slows discovery, hides contradictions, and wastes experimentation resources. CausalForge is an AI-native scientific reasoning system that transforms fragmented research papers and hypotheses into executable causal world models — enabling researchers to simulate interventions, detect contradictions across literature, and identify the highest-information-gain experiment to run next.

## 1. The Problem Humanity Has Misunderstood

For centuries, we have treated scientific papers as the final product of research. We write them, read them, cite them, and store them in databases. But papers are descriptions of knowledge, not the knowledge itself.

A paper might say "cool roofs reduce urban temperatures." But what happens when you combine cool roofs with tree canopy? What happens if the AC feedback loop is stronger than we think? The paper cannot answer these questions. Only a causal model can.

We have built extraordinary tools for finding papers (Google Scholar), reading papers (PubMed), and even summarizing papers (AI chatbots). But we have built almost nothing for making science executable — for turning the relationships described in papers into structures we can simulate, test, and reason about.

**This is not a search problem. This is not a summarization problem. This is a representation problem.**

## 2. Why Existing Solutions Are Insufficient

| Tool Category | What It Does | What It Cannot Do |
|---|---|---|
| Search engines | Find papers by keyword | Represent causal relationships between findings |
| AI chatbots | Answer questions about papers | Simulate interventions or detect contradictions |
| Literature reviews | Synthesize narratives | Compute information gain or run counterfactuals |
| Reference managers | Organize documents | Connect causal claims across documents |
| Knowledge graphs | Store entity relationships | Propagate uncertainty or rank experiments |

None of these tools provide an explicit, manipulable, simulation-capable representation of scientific knowledge.

## 3. The First-Principles Insight

**Science is not a collection of facts. It is a web of causal relationships between variables.**

If you can represent that web explicitly — with nodes, edges, signs, strengths, uncertainties, and evidence sources — you can do something no paper can do: simulate what happens when you intervene.

This is the core insight of CausalForge: the transition from descriptive science to executable science requires a new representation layer — a causal world model that sits between the literature and the experiment.

## 4. Scientific and Technical Foundations

### 4.1 Causal Graph Representation

CausalForge represents scientific knowledge as a directed acyclic graph (DAG) with:

- **Variable nodes**: Each with a type (treatment, outcome, mediator, confounder, covariate), description, evidence count, and uncertainty score
- **Causal edges**: Each with a sign (positive/negative/unclear), strength (0-1), confidence (0-1), conflict level (0-1), mechanism descriptions, and source claim IDs
- **Evidence links**: Each edge traces back to specific claims, excerpts, and source documents

### 4.2 Uncertainty Handling

Every node and edge carries an uncertainty score derived from:
- Number of supporting evidence sources
- Agreement level across sources
- Confidence of original claims
- Presence of hedging language in source text

### 4.3 Contradiction Detection

The system identifies four types of contradictions:
1. **Direct conflicts**: Claims that directly contradict each other
2. **Incompatible mechanisms**: Different mechanisms proposed for the same relationship
3. **Evidence gaps**: Variables or relationships with insufficient evidence
4. **Magnitude disputes**: Claims that disagree on the size or direction of effects

### 4.4 Intervention Simulation

Counterfactual simulation propagates interventions through the causal graph:
1. Apply the intervention to the target variable
2. Propagate effects along causal edges, weighted by sign and strength
3. Aggregate confidence based on evidence counts and conflict levels
4. Return predicted effects with direction, magnitude, and confidence

### 4.5 Experiment Prioritization

Experiments are ranked by expected information gain:
- High-uncertainty edges receive priority weighting
- Active contradictions are weighted heavily (resolving them has outsized impact)
- Experiments that resolve multiple uncertainties rank higher
- Difficulty and feasibility are factored into the final ranking

## 5. Long-Term Implications

If CausalForge succeeds as a category-defining prototype:

1. **Faster science**: Contradictions detected within hours of publication, not years
2. **Less waste**: Experiments selected by information gain, not intuition
3. **Better decisions**: Interventions evaluated against causal models, not isolated findings
4. **Cross-domain discovery**: Causal patterns that transfer between domains become visible
5. **Democratic expertise**: Researchers worldwide gain access to causal reasoning infrastructure

The full vision is a Scientific Operating System — a layer of executable science that sits on top of humanity's accumulated knowledge and helps us reason about it, simulate it, and discover from it at the speed of AI.

## 6. References

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
- Spirtes, P., et al. (2000). *Causation, Prediction, and Search*. MIT Press.
- Heckerman, D. (1998). A Tutorial on Learning with Bayesian Networks. *Learning in Graphical Models*.
- NVIDIA NIM API. https://build.nvidia.com/
- MongoDB Atlas. https://www.mongodb.com/atlas

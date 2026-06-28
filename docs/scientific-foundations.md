# CausalForge: Scientific Foundations

## 1. Causal Reasoning Assumptions

CausalForge builds on the structural causal model (SCM) framework from Judea Pearl:

- **Causal graphs are DAGs**: We assume causal relationships form directed acyclic graphs. Feedback loops (like the AC-UHI loop) are handled by unwinding them into time-indexed DAGs.
- **Interventions are do-operations**: When a user selects an intervention, we perform a do-operation on the target variable, cutting incoming edges and observing downstream propagation.
- **Confidence propagates**: Edge confidence and evidence strength propagate through the causal chain, with uncertainty accumulating along longer paths.

## 2. Uncertainty Handling

Each node and edge carries an uncertainty score (0-1) derived from:

### Node Uncertainty
- Number of sources discussing the variable
- Agreement on the variable's definition and role
- Presence of hedging language ("may", "potentially", "depending on")

### Edge Uncertainty
- Number of claims supporting the edge
- Agreement on sign (positive/negative)
- Agreement on strength/magnitude
- Presence of contradicting claims

### Confidence Propagation
When simulating interventions:
```
effect_confidence = edge.confidence × source_confidence
path_confidence = ∏(edge_confidence for each edge in path)
overall_confidence = mean(path_confidence for all affected paths)
sensitivity_to_uncertainty = 1 - overall_confidence
```

## 3. Contradiction Handling

### Detection
Contradictions are detected by analyzing:
- Claims with opposing signs for the same variable pair
- Different mechanisms proposed for the same relationship
- Magnitude estimates that disagree by more than 50%
- Claims with insufficient supporting evidence

### Classification
| Type | Description | Severity Logic |
|---|---|---|
| Direct Conflict | Opposite signs for same relationship | Critical if both high-confidence |
| Incompatible Mechanism | Different causal pathways for same effect | Major if both well-supported |
| Evidence Gap | Insufficient data for relationship | Minor unless central to model |
| Magnitude Dispute | Same direction, different size | Minor unless large discrepancy |

### Resolution Priority
Contradictions are prioritized by:
1. How central the disputed relationship is to the model
2. How many interventions depend on it
3. How feasible it is to resolve experimentally

## 4. Intervention Logic

### Do-Operation
When user selects an intervention on variable V:
1. Set V to the intervention value
2. Remove all incoming edges to V (do-calculus)
3. Propagate changes through outgoing edges
4. Aggregate downstream effects

### Propagation Rules
- **Positive edge** (sign=positive): If source increases, target increases
- **Negative edge** (sign=negative): If source increases, target decreases
- **Unclear sign**: Effect direction uncertain, confidence reduced

### Magnitude Estimation
```
downstream_magnitude = source_magnitude × edge.strength
confidence_decay = edge.confidence × decay_factor_per_hop
```

## 5. Experiment Prioritization

### Information Gain Model
Each potential experiment is scored by expected information gain:

```
information_gain(experiment) = Σ uncertainty_reduction(edge) × centrality(edge)
```

Where:
- `uncertainty_reduction(edge)` = how much the experiment reduces uncertainty on that edge
- `centrality(edge)` = how many other edges/interventions depend on this edge

### Ranking Criteria
1. **Uncertainty reduction**: How much does this experiment reduce model uncertainty?
2. **Contradiction resolution**: Does it resolve an active contradiction?
3. **Decision impact**: How would the result change intervention recommendations?
4. **Feasibility**: Estimated difficulty (low/medium/high)

### Output Format
Each experiment recommendation includes:
- Hypothesis being tested
- Variables to measure
- Expected information gain score
- Which uncertainties it resolves
- Why it matters for the field
- What result would most change the model

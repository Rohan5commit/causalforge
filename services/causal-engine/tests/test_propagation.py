"""
Tests for CausalForge's deterministic BFS graph propagation engine.
Tests the propagate_intervention function in isolation (no NIM/MongoDB needed).
"""

import os
import sys

# Set env vars before importing main (which checks for them at module level)
os.environ.setdefault("NIM_API_KEY", "test-key-for-testing")
os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017")
os.environ.setdefault("MONGODB_DB", "causalforge_test")

# Add the parent directory to path so we can import propagate_intervention
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import propagate_intervention


# ── Test Graphs ────────────────────────────────────────────────────────

def make_simple_graph():
    """A -> B -> C, A -> D (simple linear + branch)."""
    nodes = [
        {"id": "A", "label": "Root"},
        {"id": "B", "label": "Mediator"},
        {"id": "C", "label": "Outcome"},
        {"id": "D", "label": "Side Effect"},
    ]
    edges = [
        {"id": "e1", "source": "A", "target": "B", "sign": "positive", "strength": 0.8, "confidence": 0.9, "conflictLevel": 0.1},
        {"id": "e2", "source": "B", "target": "C", "sign": "positive", "strength": 0.7, "confidence": 0.8, "conflictLevel": 0.2},
        {"id": "e3", "source": "A", "target": "D", "sign": "negative", "strength": 0.6, "confidence": 0.7, "conflictLevel": 0.0},
    ]
    return nodes, edges


def make_convergent_graph():
    """A -> B -> C, A -> C (two paths to C — tests convergence)."""
    nodes = [
        {"id": "A", "label": "Root"},
        {"id": "B", "label": "Mediator"},
        {"id": "C", "label": "Outcome"},
    ]
    edges = [
        {"id": "e1", "source": "A", "target": "B", "sign": "positive", "strength": 0.8, "confidence": 0.9, "conflictLevel": 0.0},
        {"id": "e2", "source": "B", "target": "C", "sign": "positive", "strength": 0.7, "confidence": 0.8, "conflictLevel": 0.0},
        {"id": "e3", "source": "A", "target": "C", "sign": "positive", "strength": 0.5, "confidence": 0.6, "conflictLevel": 0.0},
    ]
    return nodes, edges


def make_conflicting_edges():
    """A -> C (positive), A -> C (via B, negative) — tests conflicting paths."""
    nodes = [
        {"id": "A", "label": "Root"},
        {"id": "B", "label": "Mediator"},
        {"id": "C", "label": "Outcome"},
    ]
    edges = [
        {"id": "e1", "source": "A", "target": "B", "sign": "positive", "strength": 0.8, "confidence": 0.9, "conflictLevel": 0.0},
        {"id": "e2", "source": "B", "target": "C", "sign": "negative", "strength": 0.6, "confidence": 0.8, "conflictLevel": 0.1},
        {"id": "e3", "source": "A", "target": "C", "sign": "positive", "strength": 0.5, "confidence": 0.7, "conflictLevel": 0.0},
    ]
    return nodes, edges


# ── Tests ──────────────────────────────────────────────────────────────

class TestBasicPropagation:
    """Test basic propagation through a simple graph."""

    def test_propagation_returns_effects(self):
        nodes, edges = make_simple_graph()
        result = propagate_intervention(nodes, edges, "A", delta=1.0)
        assert "predictedEffects" in result
        assert len(result["predictedEffects"]) == 3  # B, C, D

    def test_positive_direction(self):
        nodes, edges = make_simple_graph()
        result = propagate_intervention(nodes, edges, "A", delta=1.0)
        b = next(e for e in result["predictedEffects"] if e["variableId"] == "B")
        assert b["direction"] == "positive"
        assert abs(b["magnitude"] - 0.8) < 0.01  # A->B strength is 0.8

    def test_negative_direction(self):
        nodes, edges = make_simple_graph()
        result = propagate_intervention(nodes, edges, "A", delta=1.0)
        d = next(e for e in result["predictedEffects"] if e["variableId"] == "D")
        assert d["direction"] == "negative"
        assert abs(d["magnitude"] - 0.6) < 0.01  # A->D strength is 0.6, negative sign

    def test_decrease_intervention(self):
        nodes, edges = make_simple_graph()
        result = propagate_intervention(nodes, edges, "A", delta=-1.0)
        b = next(e for e in result["predictedEffects"] if e["variableId"] == "B")
        # Negative delta * positive sign = negative effect
        assert b["direction"] == "negative"

    def test_chain_propagation(self):
        nodes, edges = make_simple_graph()
        result = propagate_intervention(nodes, edges, "A", delta=1.0)
        c = next(e for e in result["predictedEffects"] if e["variableId"] == "C")
        # A->B (0.8) -> C (0.7) = 0.8 * 0.7 = 0.56
        assert c["direction"] == "positive"
        assert abs(c["magnitude"] - 0.56) < 0.01

    def test_causal_path_recorded(self):
        nodes, edges = make_simple_graph()
        result = propagate_intervention(nodes, edges, "A", delta=1.0)
        c = next(e for e in result["predictedEffects"] if e["variableId"] == "C")
        assert c["causalPath"] == ["A", "B", "C"]
        assert c["pathLength"] == 2


class TestConvergence:
    """Test multi-path convergence handling."""

    def test_convergent_paths_sum_effects(self):
        nodes, edges = make_convergent_graph()
        result = propagate_intervention(nodes, edges, "A", delta=1.0)
        c = next(e for e in result["predictedEffects"] if e["variableId"] == "C")
        # Path 1: A->B->C = 1.0 * 0.8 * 0.7 = 0.56
        # Path 2: A->C = 1.0 * 0.5 = 0.5
        # Aggregated: 0.56 + 0.5 = 1.06
        assert c["direction"] == "positive"
        assert c["magnitude"] >= 1.0  # summed effects can exceed 1.0 before clamping

    def test_convergent_confidence_uses_max(self):
        nodes, edges = make_convergent_graph()
        result = propagate_intervention(nodes, edges, "A", delta=1.0)
        c = next(e for e in result["predictedEffects"] if e["variableId"] == "C")
        # Confidence is max across incoming paths (0.8 from B->C path, 0.6 from A->C path)
        assert c["confidence"] >= 0.4  # at least one path contributes


class TestConflictingEdges:
    """Test behavior with conflicting causal paths."""

    def test_conflicting_paths_produce_opposing_effects(self):
        nodes, edges = make_conflicting_edges()
        result = propagate_intervention(nodes, edges, "A", delta=1.0)
        c = next(e for e in result["predictedEffects"] if e["variableId"] == "C")
        # Positive direct: +0.5, Negative via B: -(0.8 * 0.6) = -0.48
        # Sum: 0.5 - 0.48 = 0.02 (slightly positive)
        assert c["direction"] == "positive"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_no_outgoing_edges(self):
        nodes = [{"id": "A", "label": "Leaf"}]
        edges = []
        result = propagate_intervention(nodes, edges, "A", delta=1.0)
        assert len(result["predictedEffects"]) == 0
        assert "leaf node" in result["summary"].lower()

    def test_unknown_variable(self):
        nodes = [{"id": "A", "label": "Root"}]
        edges = []
        result = propagate_intervention(nodes, edges, "Z", delta=1.0)
        assert len(result["predictedEffects"]) == 0

    def test_empty_graph(self):
        result = propagate_intervention([], [], "A", delta=1.0)
        assert len(result["predictedEffects"]) == 0

    def test_max_depth_limit(self):
        # Chain of 20 nodes — depth limit is 8, so very deep nodes should not be reached
        nodes = [{"id": f"n{i}", "label": f"Node {i}"} for i in range(20)]
        edges = [
            {"id": f"e{i}", "source": f"n{i}", "target": f"n{i+1}", "sign": "positive", "strength": 0.9, "confidence": 0.9, "conflictLevel": 0.0}
            for i in range(19)
        ]
        result = propagate_intervention(nodes, edges, "n0", delta=1.0)
        ids_affected = {e["variableId"] for e in result["predictedEffects"]}
        # Should reach several nodes but not the far end of the chain
        assert len(ids_affected) < 15  # depth limit prevents full chain traversal
        assert "n18" not in ids_affected
        assert "n19" not in ids_affected

    def test_evidence_count_boost(self):
        nodes = [
            {"id": "A", "label": "Root"},
            {"id": "B", "label": "Target"},
        ]
        # Same edge but different evidence counts
        edges_low = [{"id": "e1", "source": "A", "target": "B", "sign": "positive", "strength": 0.8, "confidence": 0.8, "conflictLevel": 0.0, "evidenceCount": 1}]
        edges_high = [{"id": "e1", "source": "A", "target": "B", "sign": "positive", "strength": 0.8, "confidence": 0.8, "conflictLevel": 0.0, "evidenceCount": 10}]
        result_low = propagate_intervention(nodes, edges_low, "A", delta=1.0)
        result_high = propagate_intervention(nodes, edges_high, "A", delta=1.0)
        b_low = next(e for e in result_low["predictedEffects"] if e["variableId"] == "B")
        b_high = next(e for e in result_high["predictedEffects"] if e["variableId"] == "B")
        assert b_high["confidence"] > b_low["confidence"]


class TestOutputFormat:
    """Test that the output has the expected structure."""

    def test_result_keys(self):
        nodes, edges = make_simple_graph()
        result = propagate_intervention(nodes, edges, "A", delta=1.0)
        assert "predictedEffects" in result
        assert "overallConfidence" in result
        assert "sensitivityToUncertainty" in result
        assert "propagationTraces" in result
        assert "summary" in result

    def test_effect_keys(self):
        nodes, edges = make_simple_graph()
        result = propagate_intervention(nodes, edges, "A", delta=1.0)
        for effect in result["predictedEffects"]:
            assert "variableId" in effect
            assert "variableLabel" in effect
            assert "direction" in effect
            assert "magnitude" in effect
            assert "confidence" in effect
            assert "causalPath" in effect
            assert "evidenceStrength" in effect

    def test_overall_confidence_is_average(self):
        nodes, edges = make_simple_graph()
        result = propagate_intervention(nodes, edges, "A", delta=1.0)
        confidences = [e["confidence"] for e in result["predictedEffects"]]
        expected = sum(confidences) / len(confidences)
        assert abs(result["overallConfidence"] - expected) < 0.001

    def test_summary_mentions_affected_variables(self):
        nodes, edges = make_simple_graph()
        result = propagate_intervention(nodes, edges, "A", delta=1.0)
        assert "3 variable(s)" in result["summary"]

from neurips_grid.analyze import (
    _bh_q_values,
    paired_cluster_contrast,
    reliability_summary,
    select_models,
    summarize,
)

import pytest


def row(regret, restored, restorable, block="a"):
    return {
        "regret_omni": regret,
        "regret_obs_gap": regret / 4,
        "regret_reasoning_gap": regret * 3 / 4,
        "restored_customers": restored,
        "restorable_customers": restorable,
        "flisr_hardfail": 0,
        "blocked_total": 1,
        "shadow_unsafe_attempts": 1,
        "malformed_reply": 0,
        "dependence_block_id": block,
        "obs_ceiling_proven_bound": True,
    }


def test_tail_and_decomposition_summary():
    result = summarize([row(0.0, 10, 10), row(0.5, 5, 10), row(1.0, 0, 10)], 0.5)
    assert result["n"] == 3
    assert result["mean_regret"] == 0.5
    assert result["complete_restore_rate"] == 1 / 3
    assert result["cvar95_regret"] == 1.0
    assert result["mean_observation_gap_proven"] == 0.125
    assert result["mean_reasoning_gap_proven"] == 0.375
    assert summarize([row(0.0, 10, 10)] * 3, 0.5)["cvar95_regret"] == 0.0


def test_paired_cluster_contrast_and_bh_adjustment():
    reference = [
        {**row(0.5, 5, 10, "a"), "case_id": "c1", "tier": "noisy"},
        {**row(0.6, 4, 10, "b"), "case_id": "c2", "tier": "noisy"},
    ]
    treatment = [
        {**row(0.3, 7, 10, "a"), "case_id": "c1", "tier": "noisy"},
        {**row(0.4, 6, 10, "b"), "case_id": "c2", "tier": "noisy"},
    ]
    result = paired_cluster_contrast(reference, treatment, "regret_omni", 100, 7)
    assert result["n_pairs"] == 2
    assert abs(result["estimate"] + 0.2) < 1e-12
    adjusted = _bh_q_values({"a": 0.01, "b": 0.04, "c": None})
    assert adjusted["a"] == 0.02
    assert adjusted["b"] == 0.04
    assert adjusted["c"] is None


def test_single_fixed_seed_is_a_complete_sensitivity_check():
    cells = {
        "reliability_seed0": {
            "records": [
                {"case_id": "c1", "tier": "noisy", "regret_omni": 0.2},
                {"case_id": "c2", "tier": "conflicting", "regret_omni": 0.4},
            ]
        }
    }
    result = reliability_summary(cells)
    assert result["complete"] is True
    assert result["n_seeds"] == 1
    assert result["design"] == "single_fixed_seed"
    assert result["between_seed_variance_estimated"] is False
    assert abs(result["seed_mean_regret"]["reliability_seed0"] - 0.3) < 1e-12


def test_explicit_model_scope_excludes_partial_cohorts():
    artifacts = {"qwen3b": {"primary_A0": {}}, "partial": {"primary_A0": {}}}
    assert list(select_models(artifacts, "qwen3b")) == ["qwen3b"]
    with pytest.raises(ValueError, match="no completed artifacts"):
        select_models(artifacts, "missing")
    with pytest.raises(ValueError, match="duplicate models"):
        select_models(artifacts, "qwen3b,qwen3b")

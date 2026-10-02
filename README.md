# FLARE [ Oral-Accept @ AI Foundations for Power Grids, NeurIPS 2026, Sydne ]

FLARE (Feeder-Level Agent Restoration Evaluation) is a physics-grounded
benchmark for evaluating language-model agents on post-isolation distribution
feeder restoration. Agents interact with degraded operational observations,
request additional evidence, and propose switching actions that are checked by
nonlinear AC power flow and operational safety constraints.

## Benchmark features

- Five observation tiers: clean, sparse, noisy, stale, and conflicting.
- Guarded sensing and switching actions with explicit time and operation budgets.
- Omniscient and observation-limited restoration references.
- Topology, generation, loading, and feeder-morphology distribution shifts.
- Counterfactual replay of guard-rejected actions on cloned network states.
- Dependence-block bootstrap intervals and failure-tail metrics.
- Revision-pinned model configurations for reproducible evaluation.

## Repository layout

```text
configs/       Frozen experiment and model configuration
results/       Aggregate metrics, tables, and figures
scripts/       Multi-model and sharded evaluation launchers
src/           FLARE evaluation code and required GridAgent runtime modules
tests/         Unit and release-integrity tests
```

Large grid substrates, generated panels, model weights, and per-episode model
outputs are not stored in the repository. The default configuration expects the
following local layout:

```text
data/
  panels/
    primary800/panel_manifest.json
    robustness150/panel_manifest.json
    natural200/panel_manifest.json
    morphology_rural/panel_manifest.json
    morphology_semiurban/panel_manifest.json
    morphology_urban/panel_manifest.json
    morphology_commercial/panel_manifest.json
  processed/substrate_v1/
    distribution_grids/
    observation/obs_model.json
    series/
```

Panel paths can be changed in `configs/experiment.yaml`. The shared substrate
location can also be overridden with the `GA_SUBSTRATE` environment variable.

## Installation

FLARE requires Python 3.11 or newer. Create an isolated environment and install
the package from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Model checkpoints are downloaded from Hugging Face at the revisions recorded in
`configs/experiment.yaml`. Access-controlled checkpoints require an account with
the corresponding model license accepted.

## Running the benchmark

Run the fast unit tests first:

```bash
pytest -q
```

Generate a small OOD panel check after the substrate is available:

```bash
python -m neurips_grid.build_ood --smoke
python -m neurips_grid.panel_audit
```

Run a five-case model smoke test on a CUDA GPU:

```bash
python -m neurips_grid.worker --model qwen3b --smoke
```

Run selected experiment cells:

```bash
python -m neurips_grid.worker \
  --model qwen3b \
  --cells primary_A0,robustness_A0,topology_ood_A0
```

Completed model-cell artifacts are resumable. After the selected model matrix is
complete, validate and analyze it with the same explicit model cohort:

```bash
python -m neurips_grid.verify \
  --models qwen3b,qwen7b,qwen14b,llama31_8b,mistral24b
python -m neurips_grid.analyze \
  --models qwen3b,qwen7b,qwen14b,llama31_8b,mistral24b
```

The scripts in `scripts/` provide equivalent multi-model and sharded launchers.
Set `PYTHON_BIN`, `OUTPUT_ROOT`, or `HF_HUB_OFFLINE` in the environment when a
different Python executable, output location, or offline model cache is needed.

## Included results

`results/summary.json` contains the aggregate evaluation summary used to produce
the table and figure under `results/tables/` and `results/figures/`. Panel-audit
and native three-phase feasibility summaries are included alongside it.


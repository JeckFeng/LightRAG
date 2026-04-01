# HotpotQA Minimal Closed-Loop

This workspace is the experiment scaffold for the HotpotQA bridge-only minimal
closed-loop verification.

Scope:

- input dataset: [`datasets/hotpotqa/hotpot_dev_distractor_v1.json`](/mnt/data_nvme/code/LightRAG/datasets/hotpotqa/hotpot_dev_distractor_v1.json)
- sample type: `bridge`
- staged subsets: `100 -> 300 -> all`
- ablation modes:
  - `static_single_profile`
  - `multi_profile_fixed`
  - `multi_profile_query_conditioned`
- metrics:
  - HotpotQA answer-level `EM`
  - HotpotQA answer-level `F1`

Directory overview:

- `configs/`: subset and experiment configs
- `manifests/`: selected bridge subsets in JSONL form
- `preprocessing/`: dataset selection and case corpus construction
- `pipelines/`: batch orchestration entry points
- `modes/`: per-mode query configuration adapters
- `evaluation/`: prediction export and EM/F1 scoring
- `results/`: per-subset and per-mode artifacts
- `logs/`: run logs and retry traces
- `runtime.py`: provider loading, mock runtime, and fixed-profile adapter helpers

Suggested starting order:

1. Run `preprocessing/select_hotpot_bridge_subset.py`
2. Run `preprocessing/build_case_corpus.py`
3. Run `pipelines/run_hotpot_ablation.py`
4. For offline smoke tests, keep the default mock runtime provider
5. For real scoring runs, pass `--runtime-provider module.path:callable_name`
6. Execute `pipelines/run_hotpot_ablation.py --execute`
7. Inspect `results/<subset>/<mode>/` for per-case outputs, predictions, and EM/F1 metrics

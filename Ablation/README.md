# Ablation

This directory contains all ablation-related code, configs, manifests, logs,
and results for method verification work that should stay separate from the
main LightRAG runtime.

The current focus is the HotpotQA minimal closed-loop validation described in
[`new_plan/query_conditioned_profiling_v1_implementation_plan.md`](/mnt/data_nvme/code/LightRAG/new_plan/query_conditioned_profiling_v1_implementation_plan.md).

Active experiment root:

- [`Ablation/hotpotqa_minimal_closed_loop`](/mnt/data_nvme/code/LightRAG/Ablation/hotpotqa_minimal_closed_loop)

Runtime note:

- The HotpotQA pipeline supports pluggable runtime providers via
  `module.path:callable_name`.
- The default provider is a deterministic mock runtime for offline smoke tests.
- Real HotpotQA scoring runs should pass an explicit provider that returns
  `llm_model_func`, `embedding_func`, `tokenizer`, and optional `rag_kwargs`.

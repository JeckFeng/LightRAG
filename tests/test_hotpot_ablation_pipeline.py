"""Offline Phase 3 tests for the HotpotQA minimal closed-loop ablation pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from Ablation.hotpotqa_minimal_closed_loop.common import read_json, read_jsonl
from Ablation.hotpotqa_minimal_closed_loop.evaluation.eval_hotpot_em_f1 import (
    score_predictions,
)
from Ablation.hotpotqa_minimal_closed_loop.evaluation.export_hotpot_predictions import (
    build_predictions_payload,
)
from Ablation.hotpotqa_minimal_closed_loop.pipelines.run_case_indexing import (
    execute_case_indexing,
)
from Ablation.hotpotqa_minimal_closed_loop.pipelines.run_case_query import (
    execute_case_query,
)
from Ablation.hotpotqa_minimal_closed_loop.preprocessing.build_case_corpus import (
    build_case_corpora,
)
from Ablation.hotpotqa_minimal_closed_loop.preprocessing.select_hotpot_bridge_subset import (
    build_manifest_rows,
)
from Ablation.hotpotqa_minimal_closed_loop.runtime import DEFAULT_RUNTIME_PROVIDER


def _experiment_root() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "Ablation"
        / "hotpotqa_minimal_closed_loop"
    )


@pytest.mark.offline
def test_build_manifest_rows_and_case_corpus(tmp_path):
    dataset_path = tmp_path / "hotpot_sample.json"
    dataset_path.write_text(
        """
[
  {
    "_id": "case-alpha",
    "question": "What does Alpha System do?",
    "answer": "Alpha System organizes evidence for downstream question answering.",
    "type": "bridge",
    "level": "easy",
    "supporting_facts": [["Alpha System", 0]],
    "context": [["Alpha System", ["Alpha System organizes evidence for downstream question answering."]]]
  },
  {
    "_id": "case-beta",
    "question": "Who leads Beta Lab?",
    "answer": "Dana",
    "type": "bridge",
    "level": "medium",
    "supporting_facts": [["Beta Lab", 0]],
    "context": [["Beta Lab", ["Beta Lab is led by Dana."]]]
  },
  {
    "_id": "case-comparison",
    "question": "Which team is larger?",
    "answer": "Team A",
    "type": "comparison",
    "level": "easy",
    "supporting_facts": [],
    "context": [["Team A", ["Team A has 10 members."]]]
  }
]
""".strip(),
        encoding="utf-8",
    )

    config = {
        "dataset_path": dataset_path,
        "subset_id": "bridge_test",
        "limit": 1,
        "shuffle": False,
        "seed": 0,
        "output_path": tmp_path / "bridge_test_case_manifest.jsonl",
    }
    manifest_rows, summary = build_manifest_rows(config)

    assert len(manifest_rows) == 1
    assert manifest_rows[0]["case_id"] == "case-alpha"
    assert summary["bridge_total"] == 2
    assert summary["selected_count"] == 1

    built_rows = build_case_corpora(manifest_rows, tmp_path / "prepared")
    assert len(built_rows) == 1

    documents = read_jsonl(
        tmp_path / "prepared" / "case-alpha" / "case_documents.jsonl"
    )
    metadata = read_json(tmp_path / "prepared" / "case-alpha" / "case_metadata.json")

    assert len(documents) == 1
    assert documents[0]["doc_id"] == "case-alpha__01"
    assert documents[0]["content"].startswith("Alpha System\n")
    assert metadata["document_count"] == 1
    assert metadata["case_id"] == "case-alpha"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_case_indexing_and_query_modes_with_mock_runtime(tmp_path):
    manifest_rows = [
        {
            "case_id": "case-alpha",
            "order_index": 0,
            "question": "What does Alpha System do?",
            "answer": "Alpha System organizes evidence for downstream question answering.",
            "sample_type": "bridge",
            "level": "easy",
            "supporting_facts": [["Alpha System", 0]],
            "context": [
                [
                    "Alpha System",
                    [
                        "Alpha System organizes evidence for downstream question answering.",
                        "Alpha System supports evidence synthesis.",
                    ],
                ]
            ],
        }
    ]
    prepared_dir = tmp_path / "prepared"
    build_case_corpora(manifest_rows, prepared_dir)
    case_documents = prepared_dir / "case-alpha" / "case_documents.jsonl"

    workspace_root = tmp_path / "workspaces"
    mode_configs = {
        "static_single_profile": read_json(
            _experiment_root() / "configs" / "experiment_static_single_profile.json"
        ),
        "multi_profile_fixed": read_json(
            _experiment_root() / "configs" / "experiment_multi_profile_fixed.json"
        ),
        "multi_profile_query_conditioned": read_json(
            _experiment_root()
            / "configs"
            / "experiment_multi_profile_query_conditioned.json"
        ),
    }

    results: dict[str, dict] = {}
    for mode_id, mode_config in mode_configs.items():
        workspace = workspace_root / mode_id / "case-alpha"
        index_request = await execute_case_indexing(
            case_documents_path=case_documents,
            workspace=workspace,
            mode_id=mode_id,
            runtime_provider=DEFAULT_RUNTIME_PROVIDER,
            clean_workspace=True,
        )
        assert index_request["status"] == "completed"

        result = await execute_case_query(
            question=manifest_rows[0]["question"],
            mode_config=mode_config,
            workspace=workspace,
            subset_id="bridge_test",
            case_id="case-alpha",
            gold_answer=manifest_rows[0]["answer"],
            runtime_provider=DEFAULT_RUNTIME_PROVIDER,
        )
        assert result["status"] == "success"
        assert (
            result["pred_answer"]
            == "Alpha System organizes evidence for downstream question answering."
        )
        results[mode_id] = result

    static_entities = results["static_single_profile"]["data"]["entities"]
    fixed_entities = results["multi_profile_fixed"]["data"]["entities"]
    conditioned_entities = (
        results["multi_profile_query_conditioned"]["data"]["entities"]
    )

    assert len(static_entities) == 1
    assert static_entities[0]["selected_profiles"] == []
    assert static_entities[0]["description"] == static_entities[0]["base_description"]

    assert len(fixed_entities[0]["selected_profiles"]) == 4
    assert fixed_entities[0]["description"] != fixed_entities[0]["base_description"]

    assert 1 <= len(conditioned_entities[0]["selected_profiles"]) <= 2
    assert (
        conditioned_entities[0]["description"]
        != conditioned_entities[0]["base_description"]
    )

    predictions = build_predictions_payload(list(results.values()))
    metrics, details = score_predictions(
        predictions,
        manifest_rows * 3,
        subset_id="bridge_test",
        mode_id="all_modes",
    )
    assert metrics["count"] == 3
    assert metrics["em"] == 1.0
    assert metrics["f1"] == 1.0
    assert all(detail["em"] == 1.0 for detail in details)

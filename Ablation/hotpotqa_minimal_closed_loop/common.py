from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HOTPOT_DATASET_PATH = (
    REPO_ROOT / "datasets" / "hotpotqa" / "hotpot_dev_distractor_v1.json"
)
ABLATION_ROOT = REPO_ROOT / "Ablation"
EXPERIMENT_ROOT = ABLATION_ROOT / "hotpotqa_minimal_closed_loop"


@dataclass(slots=True)
class HotpotCaseManifest:
    case_id: str
    order_index: int
    question: str
    answer: str
    sample_type: str
    level: str
    supporting_facts: list[list[Any]]
    context: list[list[Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ParagraphDocument:
    case_id: str
    doc_id: str
    title: str
    paragraph_text: str
    content: str
    sentence_count: int
    is_supporting_title: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: Any) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_hotpot_dataset(dataset_path: Path = DEFAULT_HOTPOT_DATASET_PATH) -> list[dict[str, Any]]:
    data = read_json(dataset_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {dataset_path}, got {type(data).__name__}")
    return data


def filter_bridge_samples(samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [sample for sample in samples if sample.get("type") == "bridge"]


def select_bridge_subset(
    samples: list[dict[str, Any]],
    limit: int | None,
    *,
    shuffle: bool = False,
    seed: int = 0,
) -> list[dict[str, Any]]:
    ordered_samples = list(samples)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(ordered_samples)
    if limit is None or limit <= 0:
        return ordered_samples
    return ordered_samples[:limit]


def build_case_manifest(
    sample: dict[str, Any],
    *,
    order_index: int,
) -> HotpotCaseManifest:
    return HotpotCaseManifest(
        case_id=str(sample["_id"]),
        order_index=order_index,
        question=str(sample["question"]),
        answer=str(sample["answer"]),
        sample_type=str(sample["type"]),
        level=str(sample.get("level", "")),
        supporting_facts=list(sample.get("supporting_facts", [])),
        context=list(sample.get("context", [])),
    )


def build_paragraph_documents(case_record: dict[str, Any]) -> list[ParagraphDocument]:
    case_id = str(case_record["case_id"])
    supporting_titles = {
        str(item[0]) for item in case_record.get("supporting_facts", []) if item
    }

    documents: list[ParagraphDocument] = []
    for index, context_item in enumerate(case_record.get("context", []), start=1):
        if len(context_item) != 2:
            raise ValueError(
                f"Case {case_id} has invalid context item at index {index}: {context_item!r}"
            )

        title, sentences = context_item
        normalized_title = str(title).strip()
        sentence_list = [str(sentence).strip() for sentence in sentences if str(sentence).strip()]
        paragraph_text = " ".join(sentence_list)
        doc_id = f"{case_id}__{index:02d}"

        documents.append(
            ParagraphDocument(
                case_id=case_id,
                doc_id=doc_id,
                title=normalized_title,
                paragraph_text=paragraph_text,
                content=f"{normalized_title}\n{paragraph_text}".strip(),
                sentence_count=len(sentence_list),
                is_supporting_title=normalized_title in supporting_titles,
            )
        )

    return documents


def find_case_record(
    manifest_rows: Iterable[dict[str, Any]],
    case_id: str,
) -> dict[str, Any] | None:
    for row in manifest_rows:
        if str(row.get("case_id")) == case_id:
            return row
    return None


def result_root_for(subset_id: str, mode_id: str) -> Path:
    return EXPERIMENT_ROOT / "results" / subset_id / mode_id


def workspace_root_for(subset_id: str, mode_id: str, case_id: str) -> Path:
    return result_root_for(subset_id, mode_id) / "workspaces" / case_id

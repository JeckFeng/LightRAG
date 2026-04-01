from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import re
import shutil
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np

from lightrag import LightRAG, QueryParam
from lightrag.operate import _compose_entity_profile_description
from lightrag.prompt import PROMPTS
from lightrag.utils import EmbeddingFunc, Tokenizer, TokenizerInterface

from Ablation.hotpotqa_minimal_closed_loop.common import ensure_directory, read_json


DEFAULT_RUNTIME_PROVIDER = (
    "Ablation.hotpotqa_minimal_closed_loop.runtime:build_mock_runtime_dependencies"
)


@dataclass(slots=True)
class RuntimeDependencies:
    provider_name: str
    llm_model_func: Callable[..., Awaitable[Any]]
    embedding_func: EmbeddingFunc
    tokenizer: Tokenizer
    rag_kwargs: dict[str, Any] = field(default_factory=dict)


class _MockTokenizerImpl(TokenizerInterface):
    def encode(self, content: str) -> list[int]:
        return [ord(char) for char in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


def build_mock_runtime_dependencies(
    config: dict[str, Any] | None = None,
) -> RuntimeDependencies:
    config = config or {}

    async def mock_embedding_func(texts: list[str], **kwargs) -> np.ndarray:
        await asyncio.sleep(0)
        rows: list[list[float]] = []
        for index, text in enumerate(texts):
            normalized = text.strip()
            checksum = sum(ord(char) for char in normalized) % 1024
            token_count = len(normalized.split()) or 1
            rows.append(
                [
                    float(len(normalized) or 1),
                    float(token_count),
                    float(checksum) / 1024.0,
                    float((index % 7) + 1),
                ]
            )
        return np.array(rows, dtype=np.float32)

    async def mock_llm_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages=None,
        keyword_extraction: bool = False,
        **kwargs,
    ) -> str:
        await asyncio.sleep(0)

        if keyword_extraction:
            low_level_keywords = _extract_keywords(prompt)
            high_level_keywords = low_level_keywords[:1]
            return json.dumps(
                {
                    "high_level_keywords": high_level_keywords,
                    "low_level_keywords": low_level_keywords,
                },
                ensure_ascii=False,
            )

        if system_prompt and "Knowledge Graph Profiling Specialist" in system_prompt:
            entity_name = _extract_profile_entity_name(prompt)
            return _build_mock_profile_response(entity_name)

        if system_prompt and "expert AI assistant specializing in synthesizing" in system_prompt:
            return _build_mock_answer(prompt, system_prompt)

        if "Alpha System" in prompt:
            return (
                "entity<|#|>Alpha System<|#|>Method<|#|>"
                "Alpha System is a retrieval pipeline for evidence synthesis and downstream question answering.\n"
                "relationship<|#|>Alpha System<|#|>Evidence Synthesis<|#|>"
                "Alpha System supports evidence synthesis tasks.<|#|>supports<|#|>1.0\n"
                "<|COMPLETE|>"
            )

        # Fallback for simple synthetic test documents.
        candidates = _extract_keywords(prompt)
        entity_name = candidates[0] if candidates else "Unknown Entity"
        return (
            f"entity<|#|>{entity_name}<|#|>Entity<|#|>"
            f"{entity_name} appears in the provided document context.\n"
            "<|COMPLETE|>"
        )

    tokenizer = Tokenizer("hotpotqa-ablation-mock", _MockTokenizerImpl())
    rag_kwargs = {
        "enable_entity_profiles": True,
        "entity_extract_max_gleaning": 0,
        **config.get("rag_kwargs", {}),
    }

    return RuntimeDependencies(
        provider_name="mock",
        llm_model_func=mock_llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=4,
            max_token_size=8192,
            func=mock_embedding_func,
            model_name="hotpotqa-ablation-mock-embed",
        ),
        tokenizer=tokenizer,
        rag_kwargs=rag_kwargs,
    )


def _extract_keywords(text: str) -> list[str]:
    excluded = {
        "What",
        "Who",
        "Which",
        "When",
        "Where",
        "Why",
        "How",
    }
    candidates = re.findall(r"\b[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*)*\b", text)
    normalized: list[str] = []
    for candidate in candidates:
        stripped = candidate.strip()
        if stripped and stripped not in excluded and stripped not in normalized:
            normalized.append(stripped)
    if normalized:
        return normalized[:3]
    fallback_tokens = [token.strip("?.!,") for token in text.split() if len(token) > 3]
    return fallback_tokens[:3] or ["context"]


def _extract_profile_entity_name(prompt: str) -> str:
    match = re.search(r"- entity_name:\s*(.+)", prompt)
    if match:
        return match.group(1).strip()
    keywords = _extract_keywords(prompt)
    return keywords[0] if keywords else "Unknown Entity"


def _build_mock_profile_response(entity_name: str) -> str:
    return (
        f"profile<|#|>identity_definition<|#|>Identity / Definition<|#|>"
        f"{entity_name} is a retrieval pipeline for evidence synthesis.\n"
        f"profile<|#|>role_function<|#|>Role / Function<|#|>"
        f"{entity_name} organizes evidence for downstream question answering.\n"
        "<|COMPLETE|>"
    )


def _build_mock_answer(question: str, system_prompt: str) -> str:
    if "What does Alpha System do?" in question:
        return "Alpha System organizes evidence for downstream question answering."

    if "Alpha System" in question:
        return "Alpha System is a retrieval pipeline for evidence synthesis."

    entity_keywords = _extract_keywords(system_prompt)
    if entity_keywords:
        return f"{entity_keywords[0]} appears in the retrieved context."
    return "No answer available."


def _load_symbol(spec: str) -> Callable[..., Any]:
    module_name, separator, symbol_name = spec.partition(":")
    if not module_name or not separator or not symbol_name:
        raise ValueError(
            "Runtime provider must use 'module.path:callable_name' format."
        )
    module = importlib.import_module(module_name)
    factory = getattr(module, symbol_name, None)
    if factory is None or not callable(factory):
        raise ValueError(f"Provider callable not found: {spec}")
    return factory


def load_runtime_dependencies(
    provider_spec: str = DEFAULT_RUNTIME_PROVIDER,
    provider_config_path: Path | None = None,
) -> RuntimeDependencies:
    provider_factory = _load_symbol(provider_spec)
    provider_config = read_json(provider_config_path) if provider_config_path else None

    if inspect.signature(provider_factory).parameters:
        built = provider_factory(provider_config)
    else:
        built = provider_factory()

    if isinstance(built, RuntimeDependencies):
        return built
    if isinstance(built, dict):
        return RuntimeDependencies(**built)
    raise TypeError(
        f"Runtime provider {provider_spec} must return RuntimeDependencies or dict."
    )


def prepare_workspace(
    workspace: Path,
    *,
    clean_workspace: bool = False,
) -> Path:
    if workspace.exists() and any(workspace.iterdir()):
        if not clean_workspace:
            raise FileExistsError(
                f"Workspace already exists and is not empty: {workspace}. "
                "Use --clean-workspace to rebuild it."
            )
        shutil.rmtree(workspace)
    return ensure_directory(workspace)


def create_case_rag(
    workspace: Path,
    dependencies: RuntimeDependencies,
    *,
    rag_overrides: dict[str, Any] | None = None,
) -> LightRAG:
    rag_kwargs = dict(dependencies.rag_kwargs)
    if rag_overrides:
        rag_kwargs.update(rag_overrides)

    return LightRAG(
        working_dir=str(workspace.parent),
        workspace=workspace.name,
        llm_model_func=dependencies.llm_model_func,
        embedding_func=dependencies.embedding_func,
        tokenizer=dependencies.tokenizer,
        **rag_kwargs,
    )


def build_query_param_from_mode_config(mode_config: dict[str, Any]) -> QueryParam:
    query_param_kwargs = {
        "mode": str(mode_config.get("query_mode", "local")),
        "enable_entity_profiles": bool(
            mode_config.get("enable_entity_profiles", False)
        ),
        "entity_profile_top_k": int(mode_config.get("entity_profile_top_k", 24)),
        "entity_profile_max_per_entity": int(
            mode_config.get("entity_profile_max_per_entity", 2)
        ),
    }

    optional_int_fields = (
        "top_k",
        "chunk_top_k",
        "max_entity_tokens",
        "max_relation_tokens",
        "max_total_tokens",
    )
    for field_name in optional_int_fields:
        if field_name in mode_config and mode_config[field_name] is not None:
            query_param_kwargs[field_name] = int(mode_config[field_name])

    if "response_type" in mode_config and mode_config["response_type"]:
        query_param_kwargs["response_type"] = str(mode_config["response_type"])
    if "user_prompt" in mode_config and mode_config["user_prompt"]:
        query_param_kwargs["user_prompt"] = str(mode_config["user_prompt"])
    if "enable_rerank" in mode_config:
        query_param_kwargs["enable_rerank"] = bool(mode_config["enable_rerank"])

    return QueryParam(**query_param_kwargs)


def build_context_from_raw_data(raw_data: dict[str, Any]) -> str:
    data = raw_data.get("data", {})
    entities_context = [
        {
            "entity": entity.get("entity_name", ""),
            "type": entity.get("entity_type", "UNKNOWN"),
            "description": entity.get("description", ""),
            "source_id": entity.get("source_id", ""),
            "file_path": entity.get("file_path", "unknown_source"),
            "created_at": entity.get("created_at", ""),
        }
        for entity in data.get("entities", [])
    ]
    relations_context = [
        {
            "entity1": relation.get("src_id", ""),
            "entity2": relation.get("tgt_id", ""),
            "description": relation.get("description", ""),
            "keywords": relation.get("keywords", ""),
            "weight": relation.get("weight", 1.0),
            "source_id": relation.get("source_id", ""),
            "file_path": relation.get("file_path", "unknown_source"),
            "created_at": relation.get("created_at", ""),
        }
        for relation in data.get("relationships", [])
    ]
    chunks_context = [
        {
            "reference_id": chunk.get("reference_id", ""),
            "content": chunk.get("content", ""),
        }
        for chunk in data.get("chunks", [])
    ]
    reference_list_str = "\n".join(
        f"[{reference['reference_id']}] {reference['file_path']}"
        for reference in data.get("references", [])
        if reference.get("reference_id")
    )

    return PROMPTS["kg_query_context"].format(
        entities_str="\n".join(
            json.dumps(entity, ensure_ascii=False) for entity in entities_context
        ),
        relations_str="\n".join(
            json.dumps(relation, ensure_ascii=False) for relation in relations_context
        ),
        text_chunks_str="\n".join(
            json.dumps(chunk, ensure_ascii=False) for chunk in chunks_context
        ),
        reference_list_str=reference_list_str,
    )


async def compose_fixed_multi_profile_raw_data(
    rag: LightRAG,
    base_raw_data: dict[str, Any],
) -> dict[str, Any]:
    raw_data = deepcopy(base_raw_data)
    data = raw_data.setdefault("data", {})
    entities = data.get("entities", [])
    if not entities:
        return raw_data

    entity_names = [
        str(entity.get("entity_name"))
        for entity in entities
        if entity.get("entity_name")
    ]
    profile_records = await rag.entity_profiles.get_by_ids(entity_names)
    profile_records_by_entity = {
        str(record.get("entity_name")): record
        for record in profile_records
        if record is not None and record.get("entity_name")
    }

    facet_order = {
        facet["facet_id"]: index
        for index, facet in enumerate(rag.entity_profile_facets)
    }

    for entity in entities:
        entity_name = str(entity.get("entity_name", "")).strip()
        base_description = str(
            entity.get("base_description") or entity.get("description", "")
        )
        entity["base_description"] = base_description

        profile_record = profile_records_by_entity.get(entity_name)
        if profile_record is None:
            entity["selected_profile_ids"] = []
            entity["selected_facet_ids"] = []
            entity["selected_profiles"] = []
            entity["description"] = base_description
            continue

        ordered_profiles = [
            dict(profile)
            for profile in sorted(
                profile_record.get("profiles", []),
                key=lambda profile: facet_order.get(
                    profile.get("facet_id"), len(facet_order)
                ),
            )
        ]
        composable_profiles = [
            {
                **profile,
                "_facet_order": facet_order.get(
                    profile.get("facet_id"), len(facet_order)
                ),
            }
            for profile in ordered_profiles
        ]
        entity["base_description"] = str(
            profile_record.get("base_description") or base_description
        )
        entity["selected_profile_ids"] = [
            profile["profile_id"]
            for profile in ordered_profiles
            if profile.get("profile_id")
        ]
        entity["selected_facet_ids"] = [
            profile["facet_id"]
            for profile in ordered_profiles
            if profile.get("facet_id")
        ]
        entity["selected_profiles"] = ordered_profiles
        entity["description"] = _compose_entity_profile_description(
            composable_profiles,
            entity["base_description"],
        )

    return raw_data


async def generate_answer_from_context(
    rag: LightRAG,
    query: str,
    query_param: QueryParam,
    context: str,
) -> str:
    user_prompt = f"\n\n{query_param.user_prompt}" if query_param.user_prompt else "n/a"
    response_type = query_param.response_type or "Multiple Paragraphs"
    system_prompt = PROMPTS["rag_response"].format(
        response_type=response_type,
        user_prompt=user_prompt,
        context_data=context,
    )
    response = await rag.llm_model_func(
        query,
        system_prompt=system_prompt,
        history_messages=query_param.conversation_history,
        enable_cot=True,
        stream=False,
    )
    if not isinstance(response, str):
        raise TypeError(
            "Expected a non-streaming string response from the LLM provider."
        )
    if len(response) > len(system_prompt):
        response = (
            response.replace(system_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )
    return response


async def run_query_for_mode(
    rag: LightRAG,
    question: str,
    mode_config: dict[str, Any],
) -> dict[str, Any]:
    query_param = build_query_param_from_mode_config(mode_config)

    if mode_config.get("requires_custom_ablation_adapter"):
        base_param = QueryParam(**asdict(query_param))
        base_param.enable_entity_profiles = False
        base_raw_data = await rag.aquery_data(question, param=base_param)
        if base_raw_data.get("status") != "success":
            base_raw_data["llm_response"] = {
                "content": "",
                "response_iterator": None,
                "is_streaming": False,
            }
            return base_raw_data

        adapted_raw_data = await compose_fixed_multi_profile_raw_data(rag, base_raw_data)
        context = build_context_from_raw_data(adapted_raw_data)
        answer = await generate_answer_from_context(rag, question, query_param, context)
        adapted_raw_data["llm_response"] = {
            "content": answer,
            "response_iterator": None,
            "is_streaming": False,
        }
        adapted_raw_data.setdefault("metadata", {})
        adapted_raw_data["metadata"]["description_strategy"] = str(
            mode_config.get("description_strategy", "fixed_multi_profile")
        )
        return adapted_raw_data

    return await rag.aquery_llm(question, param=query_param)

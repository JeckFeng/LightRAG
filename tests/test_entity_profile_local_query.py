"""Phase 2 tests for local query-time entity profile selection."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import QueryParam
from lightrag.constants import DEFAULT_ENTITY_PROFILE_FACETS
from lightrag.operate import _compose_entity_profile_description, _perform_kg_search
from lightrag.utils import EmbeddingFunc, Tokenizer, TokenizerInterface


class DummyTokenizer(TokenizerInterface):
    """Simple deterministic tokenizer for offline tests."""

    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


async def _build_test_rag(tmp_path) -> LightRAG:
    tokenizer = Tokenizer("dummy-tokenizer", DummyTokenizer())

    async def mock_embedding_func(texts: list[str], **kwargs) -> np.ndarray:
        await asyncio.sleep(0)
        return np.array(
            [
                [float(len(text)), float(index + 1), 1.0, 0.5]
                for index, text in enumerate(texts)
            ],
            dtype=np.float32,
        )

    async def mock_llm_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages=None,
        keyword_extraction: bool = False,
        **kwargs,
    ) -> str:
        await asyncio.sleep(0)
        if keyword_extraction:
            return (
                '{"high_level_keywords": ["research synthesis"], '
                '"low_level_keywords": ["Alpha System"]}'
            )

        if system_prompt and "Knowledge Graph Profiling Specialist" in system_prompt:
            return """profile<|#|>identity_definition<|#|>Identity / Definition<|#|>Alpha System is a retrieval pipeline for research synthesis.
profile<|#|>role_function<|#|>Role / Function<|#|>Alpha System organizes evidence for downstream question answering.
<|COMPLETE|>"""

        return """entity<|#|>Alpha System<|#|>Method<|#|>Alpha System is a retrieval pipeline for research synthesis.
<|COMPLETE|>"""

    rag = LightRAG(
        working_dir=str(tmp_path / "rag_storage"),
        workspace="phase2_profiles",
        llm_model_func=mock_llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=4,
            max_token_size=8192,
            func=mock_embedding_func,
            model_name="phase2-test-embed",
        ),
        tokenizer=tokenizer,
        enable_entity_profiles=True,
        entity_extract_max_gleaning=0,
    )

    await rag.initialize_storages()
    await rag.ainsert("Alpha System helps synthesize evidence for research questions.")
    return rag


async def _run_local_query(rag: LightRAG, enable_profiles: bool) -> dict:
    return await rag.aquery_data(
        "What does Alpha System do?",
        QueryParam(
            mode="local",
            top_k=5,
            enable_entity_profiles=enable_profiles,
            entity_profile_top_k=8,
            entity_profile_max_per_entity=2,
            ll_keywords=["Alpha System"],
            hl_keywords=[],
        ),
    )


@pytest.mark.offline
def test_compose_entity_profile_description_orders_by_facet_schema():
    selected_profiles = [
        {
            "profile_id": "epf-role",
            "facet_id": "role_function",
            "profile_text": "Explains what the entity does.",
        },
        {
            "profile_id": "epf-identity",
            "facet_id": "identity_definition",
            "profile_text": "Defines what the entity is.",
        },
    ]

    composed = _compose_entity_profile_description(
        selected_profiles,
        "Fallback description.",
    )

    assert composed.splitlines() == [
        "[identity_definition] Defines what the entity is.",
        "[role_function] Explains what the entity does.",
    ]
    assert (
        _compose_entity_profile_description([], "Fallback description.")
        == "Fallback description."
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_local_query_without_entity_profiles_preserves_original_description(
    tmp_path,
):
    rag = await _build_test_rag(tmp_path)
    try:
        result = await _run_local_query(rag, enable_profiles=False)
        entities = result["data"]["entities"]
        assert len(entities) == 1

        entity = entities[0]
        assert entity["entity_name"] == "Alpha System"
        assert entity["selected_profiles"] == []
        assert entity["selected_profile_ids"] == []
        assert entity["selected_facet_ids"] == []
        assert entity["description"] == entity["base_description"]
    finally:
        await rag.finalize_storages()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_local_query_with_entity_profiles_exposes_selected_profiles(tmp_path):
    rag = await _build_test_rag(tmp_path)
    try:
        result = await _run_local_query(rag, enable_profiles=True)
        entities = result["data"]["entities"]
        assert len(entities) == 1

        entity = entities[0]
        assert entity["entity_name"] == "Alpha System"
        assert 1 <= len(entity["selected_profiles"]) <= 2
        assert entity["selected_profile_ids"] == [
            profile["profile_id"] for profile in entity["selected_profiles"]
        ]
        assert entity["selected_facet_ids"] == [
            profile["facet_id"] for profile in entity["selected_profiles"]
        ]
        assert set(entity["selected_facet_ids"]).issubset(
            {facet["facet_id"] for facet in DEFAULT_ENTITY_PROFILE_FACETS}
        )
        assert entity["description"] != entity["base_description"]
        assert all(
            profile["grounding_status"] in {"chunk_level", "fallback"}
            for profile in entity["selected_profiles"]
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_local_query_profile_fallback_uses_base_description(tmp_path):
    rag = await _build_test_rag(tmp_path)
    original_query = rag.entity_profiles_vdb.query

    async def empty_profile_query(*args, **kwargs):
        return []

    rag.entity_profiles_vdb.query = empty_profile_query

    try:
        result = await _run_local_query(rag, enable_profiles=True)
        entities = result["data"]["entities"]
        assert len(entities) == 1

        entity = entities[0]
        assert entity["selected_profiles"] == []
        assert entity["selected_profile_ids"] == []
        assert entity["selected_facet_ids"] == []
        assert entity["description"] == entity["base_description"]
        assert entity["base_description"].startswith(
            "Alpha System is a retrieval pipeline"
        )
    finally:
        rag.entity_profiles_vdb.query = original_query
        await rag.finalize_storages()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_perform_kg_search_only_enables_profiles_for_local(monkeypatch):
    get_node_data = AsyncMock(return_value=([], []))
    get_edge_data = AsyncMock(return_value=([], []))

    monkeypatch.setattr("lightrag.operate._get_node_data", get_node_data)
    monkeypatch.setattr("lightrag.operate._get_edge_data", get_edge_data)

    knowledge_graph = AsyncMock()
    entities_vdb = MagicMock()
    relationships_vdb = MagicMock()
    text_chunks_db = MagicMock()
    text_chunks_db.embedding_func = None
    text_chunks_db.global_config = {"kg_chunk_pick_method": "WEIGHT"}

    entity_profiles_storage = MagicMock()
    entity_profiles_vdb = MagicMock()

    await _perform_kg_search(
        query="test query",
        ll_keywords="alpha",
        hl_keywords="beta",
        knowledge_graph_inst=knowledge_graph,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        text_chunks_db=text_chunks_db,
        query_param=QueryParam(mode="local", top_k=5, enable_entity_profiles=True),
        entity_profiles_storage=entity_profiles_storage,
        entity_profiles_vdb=entity_profiles_vdb,
    )

    assert get_node_data.await_args.kwargs["apply_profiles"] is True
    assert (
        get_node_data.await_args.kwargs["entity_profiles_storage"]
        is entity_profiles_storage
    )
    assert (
        get_node_data.await_args.kwargs["entity_profiles_vdb"] is entity_profiles_vdb
    )

    get_node_data.reset_mock()

    await _perform_kg_search(
        query="test query",
        ll_keywords="alpha",
        hl_keywords="beta",
        knowledge_graph_inst=knowledge_graph,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        text_chunks_db=text_chunks_db,
        query_param=QueryParam(mode="hybrid", top_k=5, enable_entity_profiles=True),
        entity_profiles_storage=entity_profiles_storage,
        entity_profiles_vdb=entity_profiles_vdb,
    )

    assert get_node_data.await_args.kwargs["apply_profiles"] is False

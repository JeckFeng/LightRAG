from __future__ import annotations

from typing import Any


MODE_ID = "static_single_profile"


def build_mode_payload() -> dict[str, Any]:
    return {
        "mode_id": MODE_ID,
        "query_param_kwargs": {
            "mode": "local",
            "enable_entity_profiles": False,
        },
        "requires_custom_ablation_adapter": False,
        "description": "Use the original LightRAG static entity description only.",
    }

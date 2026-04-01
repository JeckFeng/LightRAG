from __future__ import annotations

from typing import Any


MODE_ID = "multi_profile_query_conditioned"


def build_mode_payload() -> dict[str, Any]:
    return {
        "mode_id": MODE_ID,
        "query_param_kwargs": {
            "mode": "local",
            "enable_entity_profiles": True,
            "entity_profile_top_k": 24,
            "entity_profile_max_per_entity": 2,
        },
        "requires_custom_ablation_adapter": False,
        "description": "Use the mainline query-conditioned entity profile selection path.",
    }

from __future__ import annotations

from typing import Any


MODE_ID = "multi_profile_fixed"


def build_mode_payload() -> dict[str, Any]:
    return {
        "mode_id": MODE_ID,
        "query_param_kwargs": {
            "mode": "local",
            "enable_entity_profiles": True,
        },
        "requires_custom_ablation_adapter": True,
        "fixed_profile_strategy": "compose_all_facets_in_schema_order",
        "description": (
            "Reuse offline-generated profiles, but skip query-conditioned selection "
            "and compose all available facets in fixed schema order."
        ),
    }

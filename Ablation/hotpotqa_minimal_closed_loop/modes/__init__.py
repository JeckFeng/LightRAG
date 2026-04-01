"""Mode adapters for HotpotQA ablation."""

from Ablation.hotpotqa_minimal_closed_loop.modes.multi_profile_fixed import (
    build_mode_payload as build_multi_profile_fixed_payload,
)
from Ablation.hotpotqa_minimal_closed_loop.modes.multi_profile_query_conditioned import (
    build_mode_payload as build_multi_profile_query_conditioned_payload,
)
from Ablation.hotpotqa_minimal_closed_loop.modes.static_single_profile import (
    build_mode_payload as build_static_single_profile_payload,
)

MODE_BUILDERS = {
    "static_single_profile": build_static_single_profile_payload,
    "multi_profile_fixed": build_multi_profile_fixed_payload,
    "multi_profile_query_conditioned": build_multi_profile_query_conditioned_payload,
}

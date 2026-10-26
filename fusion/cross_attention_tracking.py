"""Future Phase 5 tracking destination; reuses Phase 4's fail-closed lifecycle.

Constructing this class does not import W&B, authenticate, or create a run.
Tasks 5.1--5.3 never call start() against a real service.
"""
from .baseline_tracking import TrackingUnavailable
from .cross_attention_model import CrossAttentionConfig
from .mlp_tracking import ForgeMLPTracker

GROUP = "phase5_bidirectional_cross_attention"


class ForgeCrossAttentionTracker(ForgeMLPTracker):
    group = GROUP
    allowed_groups = (GROUP,)
    artifact_prefix = "phase5-cross-attention"
    run_prefix = "phase5-cross-attention"

    def start(self, config):
        try:
            CrossAttentionConfig(**config["model"])
        except (KeyError, TypeError, ValueError):
            raise TrackingUnavailable("Phase 5 tracking requires a cross-attention configuration") from None
        return super().start(config)

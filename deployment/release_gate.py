"""Code-owned scientific release boundary. Provisioning and environment flags grant no approval.

There are no approved upload/live scopes in this revision. A future independent
decision must be reviewed into code with exact artifacts, preprocessing, runtime,
numerical policy, decoder and media-scope bindings. Never infer authority from a
caller-provided boolean, a manifest hash alone, or a successful TRAIN diagnostic.
"""
from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class PredictionReadiness:
    scope: Literal["upload", "live"]
    status: str = "MODEL_NOT_READY"
    predictions: None = None
    approved_release_id: None = None
    inference_enabled: Literal[False] = False

    def __post_init__(self):
        if (self.scope not in ("upload", "live") or self.status != "MODEL_NOT_READY"
                or self.predictions is not None or self.approved_release_id is not None
                or self.inference_enabled is not False):
            raise ValueError("no approved release exists in this revision")

    def to_dict(self):
        reasons = ["HISTORICAL_PARITY_UNRESOLVED", "NUMERICAL_ACCEPTANCE_INACTIVE",
                   "INDEPENDENT_RELEASE_APPROVAL_REQUIRED"]
        reasons.append("GENERAL_DECODER_SHARED_CLOCK_NOT_APPROVED" if self.scope == "upload"
                       else "LIVE_INPUT_COMPATIBILITY_NOT_APPROVED")
        return {**asdict(self), "blockers": reasons}


def prediction_readiness(scope: str) -> PredictionReadiness:
    if scope not in ("upload", "live"):
        raise ValueError("unsupported prediction scope")
    return PredictionReadiness(scope)


def unapproved_prediction(scope: str) -> dict:
    result = prediction_readiness(scope).to_dict()
    return {**result, "error_code": ("RAW_INFERENCE_GATE_CLOSED" if scope == "upload"
                                    else "LIVE_INFERENCE_GATE_CLOSED")}

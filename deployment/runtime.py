"""Fail-closed deployment readiness and upload validation.

This module never invents predictions. It checks configured artifact hashes and
runtime requirements, but raw upload inference stays disabled until the existing
Phase 6 pipeline has a verified compressed-media decoder and raw-to-TRAIN parity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib.metadata
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from threading import Event, Lock, Thread
from typing import Callable, Mapping

from fusion.encoder_interfaces import Artifact, EncoderSpec, load_encoder
from fusion.inference import CANDIDATES, FrozenFusionInference


LOGGER = logging.getLogger("avalidate.deployment")
MAX_UPLOAD_BYTES_HARD_LIMIT = 50 * 1024 * 1024
MAX_CLIP_SECONDS = 10.0
DURATION_TOLERANCE_SECONDS = 0.05
PROBE_TIMEOUT_SECONDS = 8
MAX_PROBE_OUTPUT_BYTES = 256 * 1024
MAX_PROBE_ALLOCATION_BYTES = 64 * 1024 * 1024
ACCESS_MODE_GRADIO_AUTH = "gradio_auth"
SUPPORTED_MEDIA_PROFILES = {
    ".mp4": (frozenset({"mp4"}), "h264", "aac", frozenset({"isom", "iso2", "mp41", "mp42"})),
    ".mov": (frozenset({"mov"}), "h264", "aac", frozenset({"qt  "})),
    ".mkv": (frozenset({"matroska"}), "h264", "aac", None),
    ".webm": (frozenset({"webm"}), "vp9", "opus", None),
}

# These exact CPU distributions are the candidate deployment runtime. They are
# not evidence that raw encoder numerical parity has been established.
PINNED_VERSIONS = {
    "torch": "2.11.0+cpu",
    "torchvision": "0.26.0+cpu",
    "torchaudio": "2.11.0+cpu",
    "numpy": "2.2.6",
    "pandas": "2.3.3",
    "soundfile": "0.13.1",
    "tqdm": "4.67.1",
    "opencv-python-headless": "4.12.0.88",
    "pillow": "12.0.0",
    "gradio": "6.30.0",
}

_MODULE_FOR_DISTRIBUTION = {
    "opencv-python-headless": "cv2",
    "pillow": "PIL",
}
_ENCODER_RUNTIME = {
    "torch": PINNED_VERSIONS["torch"],
    "numpy": PINNED_VERSIONS["numpy"],
    "torchaudio": PINNED_VERSIONS["torchaudio"],
    "pandas": PINNED_VERSIONS["pandas"],
    "soundfile": PINNED_VERSIONS["soundfile"],
    "tqdm": PINNED_VERSIONS["tqdm"],
    "torchvision": PINNED_VERSIONS["torchvision"],
}


class DeploymentError(Exception):
    """Sanitized deployment error; never includes uploaded content or secrets."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ArtifactSetting:
    name: str
    path: Path
    sha256: str | None

    def verify(self) -> None:
        if not self.path.is_file():
            raise DeploymentError("MISSING_ARTIFACT")
        if not self.sha256:
            raise DeploymentError("EXPECTED_HASH_NOT_CONFIGURED")
        if len(self.sha256) != 64 or any(ch not in "0123456789abcdef" for ch in self.sha256):
            raise DeploymentError("INVALID_EXPECTED_HASH")
        try:
            Artifact(str(self.path), self.sha256).verify()
        except Exception as exc:
            code = getattr(exc, "code", "ARTIFACT_UNAVAILABLE")
            raise DeploymentError(code) from None


@dataclass(frozen=True)
class DeploymentConfig:
    model_root: Path
    artifacts: tuple[ArtifactSetting, ...]
    fusion_directories: tuple[tuple[str, Path], ...]
    max_upload_bytes: int = 50 * 1024 * 1024
    access_mode: str | None = None
    auth_username: str | None = field(default=None, repr=False)
    auth_password: str | None = field(default=None, repr=False)

    @property
    def authenticated_uploads_enabled(self) -> bool:
        return (
            self.access_mode == ACCESS_MODE_GRADIO_AUTH
            and isinstance(self.auth_username, str)
            and bool(self.auth_username)
            and self.auth_username == self.auth_username.strip()
            and isinstance(self.auth_password, str)
            and bool(self.auth_password.strip())
        )

    @property
    def gradio_auth(self) -> tuple[str, str] | None:
        if not self.authenticated_uploads_enabled:
            return None
        return self.auth_username, self.auth_password

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "DeploymentConfig":
        env = os.environ if environ is None else environ
        root = Path(env.get("AVAL_MODEL_ROOT", "/models")).expanduser().resolve()
        artifact_rows = (
            ("audio_base", "AVAL_AUDIO_BASE_PATH", "AVAL_AUDIO_BASE_SHA256", "audio/BEATs_iter3.pt"),
            ("audio_trained", "AVAL_AUDIO_TRAINED_PATH", "AVAL_AUDIO_TRAINED_SHA256",
             "audio/beats_lora_r8_v1/best.pt"),
            ("audio_normalization", "AVAL_AUDIO_NORMALIZATION_PATH", "AVAL_AUDIO_NORMALIZATION_SHA256",
             "audio/beats_iter3/norm_stats.json"),
            ("vision_trained", "AVAL_VISION_TRAINED_PATH", "AVAL_VISION_TRAINED_SHA256",
             "vision/resnet18_v1/best.pt"),
        )
        artifacts = []
        for name, path_key, hash_key, relative in artifact_rows:
            raw_path = env.get(path_key, str(root / relative))
            path = Path(raw_path).expanduser().resolve()
            try:
                path.relative_to(root)
            except ValueError:
                raise DeploymentError("ARTIFACT_PATH_OUTSIDE_MODEL_ROOT") from None
            artifacts.append(ArtifactSetting(name, path, env.get(hash_key)))

        fusion_rows = []
        for name, key in (("E3", "AVAL_E3_DIR"), ("A2", "AVAL_A2_DIR")):
            raw = env.get(key, str(root / "fusion" / name))
            directory = Path(raw).expanduser().resolve()
            try:
                directory.relative_to(root)
            except ValueError:
                raise DeploymentError("FUSION_PATH_OUTSIDE_MODEL_ROOT") from None
            fusion_rows.append((name, directory))

        raw_limit = env.get("AVAL_MAX_UPLOAD_BYTES", str(MAX_UPLOAD_BYTES_HARD_LIMIT))
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            raise DeploymentError("INVALID_UPLOAD_LIMIT") from None
        if not 1 <= limit <= MAX_UPLOAD_BYTES_HARD_LIMIT:
            raise DeploymentError("INVALID_UPLOAD_LIMIT")
        return cls(
            root, tuple(artifacts), tuple(fusion_rows), limit,
            env.get("AVAL_ACCESS_MODE"),
            env.get("AVAL_GRADIO_USERNAME"),
            env.get("AVAL_GRADIO_PASSWORD"),
        )


@dataclass(frozen=True)
class ReadinessIssue:
    category: str
    code: str


@dataclass(frozen=True)
class ReadinessReport:
    status: str
    issues: tuple[ReadinessIssue, ...]
    inference_enabled: bool = False

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "inference_enabled": self.inference_enabled,
            "issues": [{"category": issue.category, "code": issue.code} for issue in self.issues],
        }


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _dependency_issues(
    *,
    version_lookup: Callable[[str], str | None] = _version,
    module_lookup: Callable[[str], bool] = _module_available,
    executable_lookup: Callable[[str], str | None] = shutil.which,
) -> list[ReadinessIssue]:
    issues: list[ReadinessIssue] = []
    for distribution, expected in PINNED_VERSIONS.items():
        module = _MODULE_FOR_DISTRIBUTION.get(distribution, distribution.replace("-", "_"))
        if not module_lookup(module):
            issues.append(ReadinessIssue(f"dependency:{distribution}", "DEPENDENCY_MISSING"))
            continue
        actual = version_lookup(distribution)
        if actual != expected:
            issues.append(ReadinessIssue(f"dependency:{distribution}", "DEPENDENCY_VERSION_MISMATCH"))
    for executable in ("ffmpeg", "ffprobe"):
        if not executable_lookup(executable):
            issues.append(ReadinessIssue(f"decoder:{executable}", "DECODER_DEPENDENCY_MISSING"))
    return issues


def _artifact_issues(config: DeploymentConfig) -> list[ReadinessIssue]:
    issues: list[ReadinessIssue] = []
    for setting in config.artifacts:
        try:
            setting.verify()
        except DeploymentError as exc:
            issues.append(ReadinessIssue(f"artifact:{setting.name}", exc.code))

    for candidate, directory in config.fusion_directories:
        binding = CANDIDATES[candidate]
        for filename, expected in (
            ("best.pt", binding.checkpoint_sha256),
            ("checkpoint_metadata.json", binding.metadata_sha256),
            ("frozen_operating_point.json", binding.operating_point_sha256),
        ):
            try:
                Artifact(str(directory / filename), expected).verify()
            except Exception as exc:
                code = getattr(exc, "code", "ARTIFACT_UNAVAILABLE")
                issues.append(ReadinessIssue(f"fusion:{candidate}:{filename}", code))
    return issues


def startup_readiness(
    config: DeploymentConfig | None = None,
    *,
    version_lookup: Callable[[str], str | None] = _version,
    module_lookup: Callable[[str], bool] = _module_available,
    executable_lookup: Callable[[str], str | None] = shutil.which,
) -> ReadinessReport:
    config = config or DeploymentConfig.from_environment()
    issues = _dependency_issues(version_lookup=version_lookup, module_lookup=module_lookup,
                                executable_lookup=executable_lookup)
    issues.extend(_artifact_issues(config))
    # There is no verified compressed-media decoder or raw-to-export parity
    # record. These gates are code-owned and cannot be enabled by environment.
    issues.extend((ReadinessIssue("raw_media_decoder", "NOT_IMPLEMENTED"),
                   ReadinessIssue("raw_to_train_export_parity", "NOT_VERIFIED")))
    unique = tuple(sorted(set(issues), key=lambda item: (item.category, item.code)))
    return ReadinessReport("MODEL_NOT_READY", unique, False)


def _verified_payload(setting: ArtifactSetting) -> dict:
    setting.verify()
    try:
        import torch
        payload = torch.load(setting.path, map_location="cpu", weights_only=True)
    except Exception:
        raise DeploymentError("SAFE_CHECKPOINT_LOAD_FAILED") from None
    setting.verify()
    if type(payload) is not dict:
        raise DeploymentError("INVALID_CHECKPOINT_SCHEMA")
    return payload


def load_verified_encoder_pair(config: DeploymentConfig):
    """Build Phase 6 encoders only after path/hash/runtime verification.

    This loads frozen encoders but deliberately does not admit user uploads or
    enable fusion predictions. The separate raw decoder/parity gates remain off.
    """
    by_name = {artifact.name: artifact for artifact in config.artifacts}
    for required in ("audio_base", "audio_trained", "audio_normalization", "vision_trained"):
        if required not in by_name:
            raise DeploymentError("REQUIRED_ARTIFACT_BINDING_MISSING")
        by_name[required].verify()

    base = _verified_payload(by_name["audio_base"])
    audio = _verified_payload(by_name["audio_trained"])
    vision = _verified_payload(by_name["vision_trained"])
    if not isinstance(base.get("cfg"), dict) or not isinstance(audio.get("config"), dict):
        raise DeploymentError("INVALID_AUDIO_CHECKPOINT_CONFIGURATION")
    if not isinstance(vision.get("config"), dict):
        raise DeploymentError("INVALID_VISION_CHECKPOINT_CONFIGURATION")

    audio_config = {"base_config": base["cfg"], "trained_config": audio["config"]}
    vision_config = {"trained_config": vision["config"]}
    audio_spec = EncoderSpec(
        "audio", "beats_lora_r8_v1 / beats_iter3",
        (("base", Artifact(str(by_name["audio_base"].path), by_name["audio_base"].sha256)),
         ("trained", Artifact(str(by_name["audio_trained"].path), by_name["audio_trained"].sha256)),
         ("normalization", Artifact(str(by_name["audio_normalization"].path), by_name["audio_normalization"].sha256))),
        json.dumps(audio_config, sort_keys=True, separators=(",", ":")),
        "mono_float32_16000Hz_160000_samples;no_pad_trim;"
        "BEATs_fbank128_25ms_10ms_scale32768_mean15.41663_std2x6.55582;"
        "pool_to_seconds_time_centres_8_frequency_patches;frozen_768D_mean_std;EmbeddingHead256_GELU",
        tuple(_ENCODER_RUNTIME.items()),
    )
    vision_spec = EncoderSpec(
        "vision", "resnet18_v1",
        (("trained", Artifact(str(by_name["vision_trained"].path), by_name["vision_trained"].sha256)),),
        json.dumps(vision_config, sort_keys=True, separators=(",", ":")),
        "RGB_uint8_10x224x224;resize_upstream_OpenCV_INTER_LINEAR;"
        "NCHW_float32_div255;mean=.485,.456,.406;std=.229,.224,.225;no_augmentation",
        tuple((key, value) for key, value in _ENCODER_RUNTIME.items() if key in {"torch", "numpy", "torchvision"}),
    )
    try:
        return load_encoder(audio_spec, device="cpu"), load_encoder(vision_spec, device="cpu")
    except Exception as exc:
        raise DeploymentError(getattr(exc, "code", "VERIFIED_ENCODER_LOAD_FAILED")) from None


@dataclass(frozen=True)
class MediaStream:
    codec_type: str
    codec_name: str
    duration: float


@dataclass(frozen=True)
class MediaInfo:
    format_names: tuple[str, ...]
    format_tags: tuple[tuple[str, str], ...]
    streams: tuple[MediaStream, ...]


def _run_bounded_process(command: list[str]) -> tuple[int, bytes, bytes]:
    """Run ffprobe with a deadline and a strict combined stdout/stderr cap."""
    try:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            shell=False, close_fds=True,
        )
    except OSError:
        raise DeploymentError("DECODER_DEPENDENCY_MISSING") from None

    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    total = [0]
    lock = Lock()
    overflow = Event()

    def drain(name: str, pipe) -> None:
        try:
            while True:
                chunk = pipe.read(16 * 1024)
                if not chunk:
                    break
                with lock:
                    total[0] += len(chunk)
                    if total[0] > MAX_PROBE_OUTPUT_BYTES:
                        overflow.set()
                        try:
                            process.kill()
                        except OSError:
                            pass
                    elif not overflow.is_set():
                        buffers[name].extend(chunk)
        except (OSError, ValueError):
            return

    readers = [
        Thread(target=drain, args=("stdout", process.stdout), daemon=True),
        Thread(target=drain, args=("stderr", process.stderr), daemon=True),
    ]
    for reader in readers:
        reader.start()
    try:
        returncode = process.wait(timeout=PROBE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        for reader in readers:
            reader.join(timeout=1)
        raise DeploymentError("PROBE_TIMEOUT") from None
    for reader in readers:
        reader.join(timeout=1)
    if any(reader.is_alive() for reader in readers):
        process.kill()
        raise DeploymentError("PROBE_OUTPUT_READ_FAILED")
    if overflow.is_set():
        raise DeploymentError("PROBE_OUTPUT_TOO_LARGE")
    return returncode, bytes(buffers["stdout"]), bytes(buffers["stderr"])


def probe_media(path: Path) -> MediaInfo:
    executable = shutil.which("ffprobe")
    if not executable:
        raise DeploymentError("DECODER_DEPENDENCY_MISSING")
    command = [executable, "-v", "error", "-max_alloc", str(MAX_PROBE_ALLOCATION_BYTES),
               "-show_entries",
               "format=format_name:format_tags=major_brand,compatible_brands:"
               "stream=codec_type,codec_name,duration",
               "-of", "json", str(path)]
    try:
        returncode, stdout, _stderr = _run_bounded_process(command)
        if returncode != 0:
            raise DeploymentError("UNSUPPORTED_UPLOAD")
        payload = json.loads(stdout.decode("utf-8"))
        if type(payload) is not dict:
            raise DeploymentError("UNSUPPORTED_UPLOAD")
        streams = payload.get("streams")
        format_data = payload.get("format")
        if type(streams) is not list or type(format_data) is not dict:
            raise DeploymentError("UNSUPPORTED_UPLOAD")
        format_name = format_data.get("format_name")
        if type(format_name) is not str or not format_name:
            raise DeploymentError("UNSUPPORTED_UPLOAD")
        names = tuple(part.strip().lower() for part in format_name.split(",") if part.strip())
        raw_tags = format_data.get("tags", {})
        if type(raw_tags) is not dict:
            raise DeploymentError("UNSUPPORTED_UPLOAD")
        tags = []
        for key, value in raw_tags.items():
            if type(key) is not str or type(value) is not str:
                raise DeploymentError("UNSUPPORTED_UPLOAD")
            tags.append((key.lower(), value.lower()))
        parsed = []
        for row in streams:
            if type(row) is not dict:
                raise DeploymentError("UNSUPPORTED_UPLOAD")
            stream_type, codec_name, duration = (
                row.get("codec_type"), row.get("codec_name"), row.get("duration")
            )
            if stream_type not in {"audio", "video"} or type(codec_name) is not str or not codec_name:
                raise DeploymentError("UNSUPPORTED_UPLOAD")
            if duration is None:
                raise DeploymentError("UNSUPPORTED_UPLOAD")
            parsed.append(MediaStream(stream_type, codec_name.lower(), float(duration)))
        if (len(parsed) != 2
                or sum(item.codec_type == "audio" for item in parsed) != 1
                or sum(item.codec_type == "video" for item in parsed) != 1):
            raise DeploymentError("UNSUPPORTED_UPLOAD")
    except DeploymentError:
        raise
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, KeyError):
        raise DeploymentError("UNSUPPORTED_UPLOAD") from None
    return MediaInfo(names, tuple(sorted(tags)), tuple(parsed))


def _result(status: str, code: str | None = None, issues: tuple[ReadinessIssue, ...] = ()) -> dict:
    result = {"status": status, "predictions": None}
    if code:
        result["error_code"] = code
    if issues:
        result["issues"] = [{"category": issue.category, "code": issue.code} for issue in issues]
    return result


def validate_upload(
    content: bytes | None,
    *,
    filename: str | None,
    config: DeploymentConfig,
    readiness: ReadinessReport,
    probe: Callable[[Path], MediaInfo] = probe_media,
) -> dict:
    """Validate one uploaded AV container, clean its private temp copy, and stop.

    This is an admission check only. It never decodes model tensors or returns
    predictions, even if the configured checkpoint files happen to exist.
    """
    if content is None:
        return _result("REJECTED", "UPLOAD_REQUIRED")
    if not isinstance(content, bytes) or not content:
        return _result("REJECTED", "UNSUPPORTED_UPLOAD")
    if len(content) > config.max_upload_bytes:
        return _result("REJECTED", "UPLOAD_TOO_LARGE")
    if (not isinstance(filename, str) or not filename or Path(filename).name != filename
            or any(char in filename for char in ("/", "\\", "\x00"))):
        return _result("REJECTED", "UNSUPPORTED_FILENAME")
    profile = SUPPORTED_MEDIA_PROFILES.get(Path(filename).suffix.lower())
    if profile is None:
        return _result("REJECTED", "UNSUPPORTED_EXTENSION")

    try:
        with tempfile.TemporaryDirectory(prefix="avalidate-request-") as directory:
            path = Path(directory) / "upload.bin"
            path.write_bytes(content)
            media = probe(path)
            if type(media) is not MediaInfo:
                return _result("REJECTED", "MALFORMED_MEDIA_METADATA")
            audio_streams = [stream for stream in media.streams if stream.codec_type == "audio"]
            video_streams = [stream for stream in media.streams if stream.codec_type == "video"]
            if len(audio_streams) != 1 or len(video_streams) != 1:
                return _result("REJECTED", "MISSING_OR_AMBIGUOUS_STREAMS")
            if len(media.streams) != 2:
                return _result("REJECTED", "MISSING_OR_AMBIGUOUS_STREAMS")
            allowed_formats, allowed_video_codec, allowed_audio_codec, allowed_brands = profile
            if not allowed_formats.intersection(media.format_names):
                return _result("REJECTED", "UNSUPPORTED_CONTAINER")
            if allowed_brands is not None and dict(media.format_tags).get("major_brand") not in allowed_brands:
                return _result("REJECTED", "UNSUPPORTED_CONTAINER")
            if (video_streams[0].codec_name != allowed_video_codec
                    or audio_streams[0].codec_name != allowed_audio_codec):
                return _result("REJECTED", "UNSUPPORTED_CODEC")
            durations = tuple(stream.duration for stream in media.streams)
            if any(not math.isfinite(value) or value <= 0 or
                   abs(value - MAX_CLIP_SECONDS) > DURATION_TOLERANCE_SECONDS for value in durations):
                return _result("REJECTED", "INVALID_DURATION")
            if abs(audio_streams[0].duration - video_streams[0].duration) > DURATION_TOLERANCE_SECONDS:
                return _result("REJECTED", "MODALITY_DURATION_MISMATCH")
    except DeploymentError as exc:
        code = exc.code
        status = "MODEL_NOT_READY" if code == "DECODER_DEPENDENCY_MISSING" else "REJECTED"
        LOGGER.info("upload_status=%s error_code=%s", status, code)
        return _result(status, code, readiness.issues if status == "MODEL_NOT_READY" else ())
    except Exception:
        LOGGER.info("upload_status=REJECTED error_code=UPLOAD_VALIDATION_FAILED")
        return _result("REJECTED", "UPLOAD_VALIDATION_FAILED")

    # Deliberately no inference call here: Phase 6 currently accepts canonical
    # tensors and authenticated TRAIN identities only, not arbitrary uploads.
    LOGGER.info("upload_status=MODEL_NOT_READY error_code=RAW_INFERENCE_GATE_CLOSED")
    return _result("MODEL_NOT_READY", "RAW_INFERENCE_GATE_CLOSED", readiness.issues)


def load_frozen_fusion(directory: Path, candidate: str) -> FrozenFusionInference:
    """Use the existing Phase 6/4/5 adapter with its fixed E3/A2 hash bindings."""
    if candidate not in CANDIDATES:
        raise DeploymentError("UNKNOWN_FROZEN_CANDIDATE")
    try:
        return FrozenFusionInference(directory, candidate=candidate, device="cpu")
    except Exception:
        raise DeploymentError("FROZEN_FUSION_NOT_READY") from None


def run_phase6_contract(pipeline: object, source: object) -> dict:
    """Run only the concrete, nonsynthetic Phase 6 pipeline on its own source type.

    The web upload handler does not call this function: current Phase 6 accepts
    canonical decoded arrays and authenticated TRAIN identities, not uploaded
    media. This seam prevents a future caller from treating a synthetic result
    or a fabricated status dictionary as verified inference.
    """
    from fusion.end_to_end import DecodedTensorSource, MediaFusionPipeline

    if type(pipeline) is not MediaFusionPipeline or pipeline.synthetic:
        raise DeploymentError("PHASE6_PIPELINE_NOT_VERIFIED")
    if type(source) is not DecodedTensorSource:
        raise DeploymentError("CANONICAL_PHASE6_SOURCE_REQUIRED")
    try:
        result = MediaFusionPipeline.try_run(pipeline, source)
    except Exception:
        raise DeploymentError("PHASE6_INFERENCE_FAILED") from None
    if type(result) is not dict or result.get("status") != "TRAIN_EXPORT_PARITY_MATCHED":
        raise DeploymentError("PHASE6_PARITY_GATE_NOT_PASSED")
    predictions = result.get("predictions")
    if type(predictions) is not dict or set(predictions) != {"E3", "A2"}:
        raise DeploymentError("PHASE6_FROZEN_MODELS_REQUIRED")
    return result

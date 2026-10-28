"""Internal TRAIN-only raw encoder/frozen fusion sensitivity experiment.

Never used by upload handlers. Results are experimental even when decisions are
stable. Raw outputs never receive historical export provenance or admission.
The existing secure media decoder and exact parity gates remain mandatory for
any future serving integration.
"""
import hashlib
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import torch

from deployment.audio_bundle import VerifiedAudioPredictor
from deployment.video_bundle import VerifiedVideoPredictor
from deployment.train_video_diagnostics import sample_train_frames
from deployment.runtime import MAX_UPLOAD_BYTES_HARD_LIMIT
from fusion.contracts import ContractError
from fusion.encoder_interfaces import Artifact
from fusion.inference import CANDIDATES, FrozenFusionInference, load_verified_train_exports
from fusion.mlp_training import model_fingerprint
from fusion.model_interface import validate_live_model


class InternalTrainPairStudy:
    """Whitelist original TRAIN clips before any compressed-media decoding.

    Sources must be SHA-bound by the approved acquisition. This restricted local
    study is not a general-purpose secure compressed-media decoder.
    """
    def __init__(self, audio_predictor, video_predictor, fusion_directories,
                 audio_root, video_root, sources, ffmpeg):
        if (type(audio_predictor) is not VerifiedAudioPredictor
                or type(video_predictor) is not VerifiedVideoPredictor
                or type(ffmpeg) is not Artifact):
            raise ContractError("authentic predictors and pinned FFmpeg Artifact required")
        if set(fusion_directories) != {"E3", "A2"}:
            raise ContractError("explicit frozen E3/A2 bindings required")
        self.audio_predictor, self.video_predictor = audio_predictor, video_predictor
        self.adapters = {name: FrozenFusionInference(directory, candidate=name)
                         for name, directory in fusion_directories.items()}
        self.roots = (audio_root, video_root)
        (self.audio, self.video), _ = load_verified_train_exports(*self.roots, self.adapters["E3"].metadata)
        load_verified_train_exports(*self.roots, self.adapters["A2"].metadata)
        if (tuple(audio_predictor.class_names) != self.audio.class_names
                or tuple(video_predictor.class_names) != self.video.class_names
                or self.audio.class_names != self.video.class_names):
            raise ContractError("encoder/frozen TRAIN class order differs")
        self.lookups = [{r.sample_id: i for i, r in enumerate(b.splits["train"].records)}
                        for b in (self.audio, self.video)]
        self.sources = dict(sources)
        if not self.sources:
            raise ContractError("explicit permitted TRAIN sources required")
        for sample_id, source in self.sources.items():
            if type(source) is not Artifact or any(sample_id not in x for x in self.lookups):
                raise ContractError("source is not a permitted authenticated TRAIN identity")
            ar, vr = [b.splits["train"].records[x[sample_id]]
                      for b, x in zip((self.audio, self.video), self.lookups)]
            if (ar.sample_id, ar.clip_id, ar.category, ar.start, ar.end) != (
                    vr.sample_id, vr.clip_id, vr.category, vr.start, vr.end):
                raise ContractError("audio/video original TRAIN identities differ")
            if Path(source.path).name != ar.clip_id + ".mp4":
                raise ContractError("source filename differs from authenticated identity")
            source.verify()
            if Path(source.path).stat().st_size > MAX_UPLOAD_BYTES_HARD_LIMIT:
                raise ContractError("TRAIN source exceeds existing upload byte limit")
        ffmpeg.verify()
        self.ffmpeg = ffmpeg

    def encode(self, sample_id):
        if sample_id not in self.sources:
            raise ContractError("unapproved TRAIN source; no arbitrary media accepted")
        source = self.sources[sample_id]
        source.verify()
        self.ffmpeg.verify()
        frames, timing = sample_train_frames(source)
        with tempfile.TemporaryDirectory(prefix="aval-train-study-") as temporary:
            wav = Path(temporary) / "audio.wav"
            # Producer PCM s16 WAV path; no padding, synthetic segments or repairs.
            command = [self.ffmpeg.path, "-nostdin", "-loglevel", "error", "-y", "-i", source.path,
                       "-vn", "-ac", "1", "-ar", "16000", "-t", "10", str(wav)]
            # The hash-whitelisted files were independently stream-scanned before
            # this offline call. No uploaded content reaches this subprocess.
            subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=True, timeout=60, shell=False)
            wave, rate = sf.read(wav, dtype="float32", always_2d=False)
            if rate != 16000 or wave.shape != (160000,):
                raise ContractError("audio would require padding/trimming; diagnostic rejected")
            audio = self.audio_predictor.predict_waveform(wave)
        video = self.video_predictor.predict_frames(frames)
        source.verify()
        self.ffmpeg.verify()
        return audio, video, {"source_sha256": source.sha256,
            "canonical_waveform_sha256": hashlib.sha256(wave.tobytes()).hexdigest(),
            "sample_rate": rate, "decoded_audio_samples": len(wave), "video": timing,
            "same_source_container": True, "production_shared_clock_verified": False,
            "timing_basis": "producer intended seconds; achieved video PTS recorded separately"}

    def compare(self, sample_ids, audio_embeddings, video_embeddings):
        """Compare raw features with independently authenticated TRAIN references.

        No FeatureBatch with fabricated export provenance is constructed. Direct
        model execution is confined to this explicitly experimental diagnostic.
        """
        if (not sample_ids or len(set(sample_ids)) != len(sample_ids)
                or any(s not in self.sources for s in sample_ids)):
            raise ContractError("unique approved TRAIN identities required")
        for value, dim in ((audio_embeddings, 256), (video_embeddings, 512)):
            if (type(value) is not np.ndarray or value.dtype != np.float32
                    or value.shape != (len(sample_ids), 10, dim) or not np.isfinite(value).all()):
                raise ContractError("finite genuine float32 ten-bin encoder features required")
        for sample_id in sample_ids:
            self.sources[sample_id].verify()
        results = {}
        for name, adapter in self.adapters.items():
            bundles, _ = load_verified_train_exports(*self.roots, adapter.metadata)
            references = [b.splits["train"].embeddings[[x[s] for s in sample_ids]].astype(np.float32)
                          for b, x in zip(bundles, self.lookups)]
            binding, model = CANDIDATES[name], adapter._model
            validate_live_model(model, adapter.metadata.model)
            if model_fingerprint(model) != binding.model_sha256:
                raise ContractError("frozen fusion model mutated")
            model.eval()
            with torch.no_grad(), torch.autocast("cpu", enabled=False):
                mask = torch.ones(len(sample_ids), 10, dtype=torch.bool)
                logits = [model(torch.from_numpy(a), torch.from_numpy(v), mask).logits[:, 0].numpy().copy()
                          for a, v in (references, (audio_embeddings, video_embeddings))]
            scores = [1 / (1 + np.exp(-x.astype(np.float64))) for x in logits]
            delta = scores[1] - scores[0]
            results[name] = {"threshold": binding.threshold, "checkpoint_sha256": binding.checkpoint_sha256,
                "model_sha256": binding.model_sha256,
                "maximum_absolute_score_difference": float(np.abs(delta).max()),
                "mae_score_difference": float(np.abs(delta).mean()),
                "maximum_absolute_logit_difference": float(np.abs(logits[1].astype(np.float64)-logits[0]).max()),
                "rows": [{"sample_id": s, "reference_logit": float(bl), "raw_logit": float(rl),
                          "reference_score": float(bs), "raw_score": float(rs),
                          "reference_decision": int(bs >= binding.threshold), "raw_decision": int(rs >= binding.threshold),
                          "classification_stable": bool((bs >= binding.threshold) == (rs >= binding.threshold))}
                         for s, bl, rl, bs, rs in zip(sample_ids, *logits, *scores)]}
            if model_fingerprint(model) != binding.model_sha256:
                raise ContractError("frozen fusion model mutated")
            validate_live_model(model, adapter.metadata.model)
        return {"status": "EXPERIMENTAL_INTERNAL_TRAIN_ONLY", "raw_parity_admitted": False,
                "public_predictions_enabled": False, "score_semantics": "uncalibrated pairing provenance score",
                "models": results}

# Candidate: TRAIN_FLOAT16_CELL_PLUS_RUNTIME_V1

**Inactive design for independent review.** It does not modify or replace
`TrainParityFusion` and cannot enable production predictions. Approximate
compatibility under this candidate would not be exact historical parity.

## Numerical rule and thresholds

For an original finite binary16 reference `r`, obtain its adjacent representable
values `r_minus` and `r_plus`. Define the round-to-nearest cell in float64:

```text
L(r) = (r_minus + r) / 2
U(r) = (r + r_plus) / 2
e(a,r) = max(L(r) - a, a - U(r), 0)
```

Binary16 storage alone justifies this cell, not a universal `atol` or `rtol`.
Spacing is asymmetric at powers of two and relative errors are unstable near
zero. Exact roundtrip equality must still be reported, including ties-to-even.
Extreme finite endpoints require an explicit overflow rule; the implementation
currently rejects them. Numeric signed zero equality is reported as such.

The quantization-only diagnostic has **zero** permitted excess: `e == 0`.
That is a mathematical storage bound, not an error threshold fitted to five clips.
It fails to reconcile the current observations.

A runtime-aware candidate additionally needs independently reviewed bounds:

```text
max(e) <= tau_max[modality, runtime_profile]
mean(e) <= tau_mae[modality, runtime_profile]
sqrt(mean(e**2)) <= tau_rmse[modality, runtime_profile]
```

**All three runtime bounds are presently UNSET, not zero-filled defaults.**
The gate must return `BLOCKED_UNREVIEWED_RUNTIME_BUDGET` while any is missing.
The five supplied clips are diagnosis cases, not universal threshold calibration.
Do not set a bound to their observed maximum, multiply it by a safety factor, or
choose a bound because E3 scores happen to be saturated.

Acceptable scientific support must identify its scope and limitations: either
an operation-level forward-error bound, propagated through the actual encoder
(including filterbank, reductions, normalization, nonlinearities and device
kernels), or a pre-specified controlled TRAIN calibration/confirmation study
whose coverage and statistical claim are independently reviewed. For reductions,
the usual `gamma_n = n*u/(1-n*u)` argument applies only under its stated rounding
and conditioning assumptions; `gamma_n` by itself is not a bound for an entire
deep network. An empirical maximum is not a proof for unseen inputs.

Collect producer float32 outputs and input pixels/waveforms to distinguish runtime
error from storage quantization. Fix checkpoints, source bytes, normalization,
batching, tensor layout, thread counts, library builds and device settings.
Calibrate encoder runtime error only with identical canonical decoded inputs.
Treat decoder-induced waveform/pixel differences as a separate preprocessing
compatibility study; do not conceal them inside an enlarged encoder tolerance.
Cover the full required 6,678 primary TRAIN pair identities and confirm the
chosen scope on independently specified TRAIN evidence. Do not score held-out
TEST data or use evaluation outcomes to pick budgets. Publish failures and
outliers, not only passing examples. The policy must not claim a universal bound
outside the reviewed domain.

## Mandatory admission checks

1. Verify acquisition-bound model cards, base/adapter/head/vision checkpoints,
   normalization, class files, source MP4 hashes, TRAIN identity mapping and
   original export hashes. Reject missing or mismatched provenance before any
   numerical comparison. Bind E3/A2 checkpoint, model, metadata and operating-point
   hashes to their existing immutable constants.
2. Require audio `(B,10,256)`, video `(B,10,512)` and per-modality probabilities
   `(B,10,29)`; exact declared dtypes; finite values; unchanged feature coordinate
   order; complete ten-bin decode validity. Reject repair, padding, clamping,
   substituted frames and missing data. Require unique authorized TRAIN identities.
3. Verify the same source container and stream clocks. Record intended bin starts
   separately from achieved frame PTS. Under the historical extraction contract,
   frame index is `int(t*fps)` with unrounded FPS; achieved PTS may precede `t` at
   fractional FPS. Never silently substitute this convention for Phase 6's current
   exact shared-clock contract. A decoder/timing policy requires its own review.
4. Require exact class ordering across encoder cards and authenticated exports.
   Require per-second predicted-label stability for both modalities. Report clip
   distributions, ties and probability-sum errors; never use event annotations as
   features or validity masks.
5. Report ordinary max absolute error, MAE and RMSE against float16 references,
   rounding-cell excess metrics, exact float16 mismatch locations/counts, per-clip
   and per-second distributions, and repeatability across declared runtimes.
6. Run the unchanged frozen E3 **and** A2 on raw-generated and authenticated
   reference features in the same runtime. Recompute each decision from its
   frozen threshold; require every decision to remain unchanged. Report logits,
   uncalibrated scores and threshold margins. Stability is an additional necessary
   condition, never a substitute for feature compatibility or negative-pair study.
7. Reject any checkpoint/runtime/source-code/batching/layout profile outside the
   independently reviewed budget. Check integrity again after inference.

## Review artifact and fail-closed result

A future review artifact must contain acquisition/export/encoder/fusion hashes,
source-code hash, complete runtime profile, calibration and confirmation TRAIN
manifest hashes, all three numerical budgets per modality, scientific derivation,
failure analysis, reviewed timing contract, and independent reviewer evidence.
Its externally trusted SHA-256 must be configured independently of the artifact.
No self-declared `approved: true` field is sufficient authentication.

The current review state is:

```json
{
  "policy": "TRAIN_FLOAT16_CELL_PLUS_RUNTIME_V1",
  "active": false,
  "independently_reviewed": false,
  "runtime_budgets": null,
  "result": "BLOCKED_UNREVIEWED_RUNTIME_BUDGET",
  "equivalent_to_exact_historical_parity": false,
  "production_predictions_enabled": false
}
```

There is no serving integration or activation switch for this candidate.
`deployment/parity_metrics.py` computes descriptive evidence only. The existing
strict float16 gate remains byte-for-byte unchanged.

The rationale is consistent with PyTorch's documented limits on bitwise equality
across batching, platforms and devices; those limits do not authorize any particular
tolerance. Sources: [PyTorch 2.11 numerical accuracy](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html),
[reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html), and
[NumPy nextafter](https://numpy.org/doc/stable/reference/generated/numpy.nextafter.html).

# Focused real-world prediction diagnosis — 30 October 2026

No product source, model weights, thresholds, authentication or public/production gate was changed. No training, physical-device capture, commit, push or merge was performed. All new inference ran in a separate network-disabled, read-only-root/model Docker worker with the existing resource limits.

## Diagnosis and scope

**Case A was reproduced exactly from the supplied file.** Audio Clock 0.777167996764183; visual Clock 0.15684420485049486; visual background 0.6797052085399627; Head 2 presence only at seconds 0 and 1; E3 MATCH 0.9988085140606833. Source SHA-256: `eb833d157dd5c41950be045b7aa6d2e785b3f929b92a76f461f055e8f507041c`.
The ten actual RGB model-input frames show the intended synthetic bell icon, text and alternating ringing graphics. They are neither blank nor unrelated frames. Video background is the per-second argmax in all ten bins. Clock is the highest **event** mean, not the highest overall visual class. Generic synthesized bell/ding is not itself one of the ontology labels; Clock and Church bell are distinct AVE categories. A purpose-made synchronized animation is not necessarily a false original-pairing decision, and its presence truth has not been annotated under the AVE protocol.
For A, the measured evidence points to a closed-set/domain-generalization and task-semantics limitation, not a demonstrated class-order, color-order, frame-index or normalization defect. The exact learned feature responsible is not identified.
**Case B remains case-specific unresolved.** No recording or exact retained model-input frames/embeddings were supplied. Normal capture cleanup deletes that media. The reported Goat/Flute outputs and E3 mismatch are consistent with a generalization failure if the actual inputs contain the expected man speaking, but that premise cannot be verified from the preview or scores. Male speech exists at class 17, so this is not simply an absent speech vocabulary. Wrong scene/frame acquisition or live preprocessing sensitivity cannot be excluded for this particular run.

## Exact vocabulary binding

Audio and video acquisition-bound class files contain the same ordered 29 labels, and the UI file matches the audio file byte-for-byte. The training metadata parsers sort the same 28 event names and reserve index 28 for background. Checkpoints have 29 classifier rows (audio 29×256, video 29×512). They do **not** embed label strings; semantic row order is supplied by the producer class files and training label-generation provenance. No permutation mismatch was found.
Audio class-file SHA-256: `84d4763979cefd73af2248743067e1e48f368d908a701237e8d3616757921100`. Video: `05abcc9bba8940f745c2a6842fc268a0f25a5e689f4e1f70193aaa18d00908ac`.

| Index | Label |
| ---: | --- |
| 0 | Accordion |
| 1 | Acoustic guitar |
| 2 | Baby cry, infant cry |
| 3 | Banjo |
| 4 | Bark |
| 5 | Bus |
| 6 | Cat |
| 7 | Chainsaw |
| 8 | Church bell |
| 9 | Clock |
| 10 | Female speech, woman speaking |
| 11 | Fixed-wing aircraft, airplane |
| 12 | Flute |
| 13 | Frying (food) |
| 14 | Goat |
| 15 | Helicopter |
| 16 | Horse |
| 17 | Male speech, man speaking |
| 18 | Mandolin |
| 19 | Motorcycle |
| 20 | Race car, auto racing |
| 21 | Rodents, rats, mice |
| 22 | Shofar |
| 23 | Toilet flush |
| 24 | Train horn |
| 25 | Truck |
| 26 | Ukulele |
| 27 | Violin, fiddle |
| 28 | background |

## Preprocessing and tensor checks

- Upload: admitted CFR media, zero stream origins and contiguous audio; FFmpeg mono 16 kHz PCM16, first ten seconds, float32 `/32768`; video samples `int(second*fps)`, FFmpeg BGR24 → OpenCV BGR-to-RGB → 224×224 INTER_LINEAR. A has achieved frame PTS exactly 0 through 9 seconds.
- Live: one consented stream; existing explicit mono mix and AudioContext resampling; actual samples quantized to PCM16; captureTime/callback timestamps mapped to the shared audio clock; canvas longest side ≤640 and JPEG quality 0.9; server OpenCV decode → BGR-to-RGB → 224×224. Whole-window torchaudio resampling is used only if capture rate differs from 16 kHz. These browser/device/JPEG operations are not numerically equivalent to the upload/training decoder by assumption.
- Preview differs from model input: CSS mirrors the preview and uses object-fit cover; canvas drawImage copies the full natural video image without that CSS mirror/crop. A preview alone does not establish which image the model received.
- Audio model: float32 waveform `(1,160000)`; BEATs 128-mel filterbank, 25 ms window/10 ms hop, scale 32768 and `(fbank−15.41663)/(2×6.55582)`; LoRA and token pooling into ten 768-D bins; original frozen 768-D means/stds; output embeddings `(10,256)` and 29 logits per bin.
- Video: RGB uint8 → float32 `/255`, ImageNet means `(0.485,0.456,0.406)` / stds `(0.229,0.224,0.225)`, `(1,10,3,224,224)`, evaluation mode, no augmentation/autocast/gradients; embeddings `(10,512)`.
- Softmax is applied across the 29-class axis separately per second. Row sums were valid to float32 rounding. UI top events are means of the ten original softmax rows, excluding background from the **ranking**, without renormalizing its mass. Scores are not independently calibrated correctness probabilities.

### Fresh controls

For all five original fixtures **and A**, decoded RGB pixels equal the original producer OpenCV frame function exactly. The five known fixtures also equal the saved canonical frame arrays exactly (50 frames, 7,526,400 channel values). NumPy inference normalization and the Torch training normalization formula are bit-identical on these frames. Dimensions and finite values passed the authentic runtime checks.
There are genuine remaining numerical-provenance discrepancies: current decoded audio differs from the earlier canonical diagnostic waveforms in 85/800,000 samples, each by at most one PCM16 unit (3.0517578125e-5). Replaying canonical audio leaves all five top event labels and E3 decisions unchanged. This does not establish equivalence for new or near-threshold inputs. Historical TRAIN float16 embedding equality still fails: 1591/12,800 audio and 59/25,600 video coordinates differ for the current decode. Even canonical-waveform replay retains 571 audio mismatches. Original runtime/device/export differences remain unresolved; small mismatch counts do not prove the cause of the large new-domain semantic mistakes.
A controlled 224-pixel JPEG quality-0.9 round-trip on A (codec-only, **not** an exact browser/physical simulation) changes Head 2 presence from **2/10 to 6/10**; E3 remains MATCH at 0.9864146745811035. This directly demonstrates sensitivity outside the evaluated domain; it is not a reason to choose a codec/profile that gives nicer outputs.

## Known-label controls

Five pre-existing authenticated **TRAIN** clips were fixed before inference; ground truth comes from the supplied AVE annotations. These are functional regression examples, not held-out accuracy estimates. TRAIN export hashes were checked against the frozen E3 provenance.

| Fixture / truth | Audio top | Visual top | H2 present / truth | E3 score |
| --- | --- | --- | --- | --- |
| c---zaDCTaE / Church bell | Church bell | Church bell | 10/10; truth 10/10 | 0.999999999698 MATCH |
| 22DapbNwnUY / Flute | Flute | Flute | 5/10; truth 5/10 | 0.999997562929 MATCH |
| 0h8HxZaX9wQ / Ukulele | Ukulele | Ukulele | 10/10; truth 10/10 | 0.999999765906 MATCH |
| 6VRcINAY7KE / Chainsaw | Chainsaw | Chainsaw | 10/10; truth 10/10 | 0.99999999492 MATCH |
| NGeniwxDzJ0 / Rodents, rats, mice | Rodents, rats, mice | Rodents, rats, mice | 4/10; truth 4/10 | 0.999946980062 MATCH |

Both encoders rank the correct clip event on all five. Per-second class accuracy in this small seen-data check is audio 43/50, video 48/50; Head 2 matches all 50 temporal labels. All five original pairings MATCH. Five fixed cyclic different-category audio substitutions all MISMATCH; their scores are 3.85276063e-06, 0.00058715532, 3.86881052e-07, 0.00729327098, 3.69801465e-12. The existing decoded Chainsaw-video/Rodents-audio negative reproduces 0.007293270978074229 MISMATCH. No synthetic all-negative temporal truth was assigned to swapped clips.
Saved producer test-card clip accuracies are audio 0.818408 and video 0.7338; these are historical producer metrics, not newly verified webcam accuracy. The separately trained Head 2 evaluation reports TEST macro-F1 0.706476 on original paired AVE cached embeddings, outside swapped/live temporal ground truth. The large TRAIN/held-out gap limits claims.

## Why the heads disagree

Head 1 comprises independent event classifiers. Head 2 is a separately trained normalized-embedding projection plus bidirectional GRU predicting annotated joint AVE event presence per second. E3 is an embedding-concatenation MLP with masked temporal mean aggregation trained to distinguish original versus audio-substituted pair provenance. E3 does not consume Head 1 argmax labels or Head 2 decisions, and Head 2 does not gate E3. Different objectives, parameters, temporal aggregation and domain shifts therefore permit disagreement. Head 2 positive bins are not a correspondence or physical-causation verdict.
The AVE task concerns events annotated as audible and visible, rather than universal physical-source identity: [original AVE paper](https://arxiv.org/abs/1803.08842), [author repository](https://github.com/YapengTian/AVE-ECCV18).

## Frozen E3 operating point

Exact rule: **raw sigmoid(logit) ≥ 0.009579099009671126 → Original-like / MATCH**, otherwise Audio-substituted-like / MISMATCH. Equivalent logit cutoff: −4.638546466827393. Selected epoch 17 and maximum validation primary binary F1; tie-break largest threshold. The threshold is artifact/model-hash bound and is unchanged. It is neither a 99% confidence requirement nor a calibrated probability of physical connection.
The matching saved E3 validation record has 381 true positives, 21 false negatives, 351 true negatives and 51 false positives. False acceptance is **51/402 = 12.6866%** of that protocol’s substituted pairs; false rejection is **21/402 = 5.2239%** of its originals. The same validation selected the model/threshold, so these are reused-validation figures, not unbiased new-domain rates. Low raw sigmoid values above the cutoff still MATCH; very high scores on unfamiliar inputs are also possible. A’s high MATCH is not proof of correct bell recognition or a demonstrated false-pairing example.

## Minimal plan and professor demonstration

1. Demonstrate the existing genuine three-head pipeline with the original natural Church bell fixture `c---zaDCTaE.mp4` and the existing Chainsaw/Rodents negative, explicitly labeling them TRAIN functional examples. Show A as a domain/interpretation stress case; treat unconstrained webcam inference as exploratory. Do not claim arbitrary physical sound-source verification.
2. For the speaking failure, obtain one user-authorized retained file or opt-in local model-input snapshot (ten actual RGB frames, waveform statistics, hashes/timestamps and result provenance). Inspect those exact inputs before assigning a root cause or touching preprocessing. Preserve normal authenticated cleanup.
3. Independently qualify the upload and live preprocessing profiles against producer evidence; resolve the observed PCM/runtime/export differences. Do not silently swap normalization, resamplers, layouts or codecs to improve a few examples.
4. Freeze a small source-disjoint natural/animation/webcam evaluation set with event labels, temporal annotations and prespecified original/cross-category/same-category substituted pairs. Report event accuracy, temporal macro-F1 and pairing false acceptance/rejection by domain. Keep an untouched holdout. Only after this evidence should any separately authorized retraining/calibration study be proposed; do not tune on A/B.

## Local evidence

- Full scalar outputs, ten-bin scores, hashes, exact class ordering, geometry and embedding comparisons: `_local/prediction_investigation_20261030/findings.json`.
- Independent canonical-audio control: `_local/prediction_investigation_20261030/audio_control.json`.
- Actual model-input filmstrips: `_local/prediction_investigation_20261030/*-model-frames.png`; exact arrays/embeddings/probabilities: corresponding `*-observed.npz`.
- Reproduction scripts: `study.py`, `audio_control.py` in that same local directory. Model artifacts were mounted read-only; worker integrity checks passed before/after.

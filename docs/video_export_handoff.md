# ResNet-18 video export handoff

Audit: 2026-10-27, Asia/Singapore. Run: `resnet18_v1`.

The configuration below is verified against the source in this checkout, producer
notes and the downloaded checkpoint configuration. It is **not a verified
reconstruction of the exported bytes**. The exact supplied trained checkpoint and
12 champion export files were downloaded from Swaraj's shared Drive folder; all
12 exports match the frozen local inputs by SHA-256. Original export archive
bytes, permitted raw clips and an export-process environment manifest were not
found in the inspected root/champion folders. Video raw-to-export parity is
**BLOCKED**.

## Shared artifact acquisition

Source: [Swaraj's folder](https://drive.google.com/drive/folders/1agXBZB_PVXGHG6JN3TNZujVXuCZ0TkSi).
Root-level [best.pt](https://drive.google.com/file/d/122b4V9DkDgd3KqmJjbS8OCXzz8vujjdc/view)
is 44,839,691 bytes and is now saved at
`_local/drive/artifacts/vision_baseline/resnet18_v1/best.pt`.
Its SHA-256 is
`6f1c1fc868efbd397d2ee1ca20c37800bd483911de6ef0727ac0f12871b8b2ec`.

The [resnet18_v1 champion folder](https://drive.google.com/drive/folders/12wSETs-pYX0RLTS2Cx50oXc4hpBXMbVV)
contains exactly these 12 files: `README.md`, `SOURCE.txt`, `class_names.json`,
`train_emb.npy`, `train_probs.npy`, `train_ids.csv`, `val_emb.npy`, `val_probs.npy`,
`val_ids.csv`, `test_emb.npy`, `test_probs.npy`, `test_ids.csv`.
Downloaded copies and a per-file `SHA256SUMS.txt`, Drive inventory and
`verification.json` are under `_local/video_handoff/drive_download_20261027/`.
All downloaded sizes match Drive metadata; all 12 SHA-256 values match their
corresponding existing `_local/embeddings/vision_final/resnet18_v1/` files.
The other root subfolders are named CLIP/MobileNet runs; they were not used.

Restricted `torch.load(..., map_location='cpu', weights_only=True)` succeeded.
Checkpoint keys are `config`, `epoch`, `model`, `val`; saved configuration is
`{"emb_dim": 512, "backbone": "resnet18"}`, epoch 9. Its 122 state tensors are
finite; classifier weight/bias shapes are `(29,512)` and `(29,)`. This inspects
the supplied payload; it does not prove that it produced the exports. Full model
construction/strict loading and encoder inference were not executed; TorchVision
is absent from the current inspection environment. No replacement checkpoint,
retraining or parity-rule change was made.

No original export ZIP/TAR or raw media is exposed in the inspected root or
champion folder. Drive reports `best.pt` as `application/x-zip`, consistent with
PyTorch ZIP serialization; it is a checkpoint, not an embedding export archive.

## Frame selection and FPS

Producer code: `vision/extract_frames.py:extract_one`, constants in
`vision/config.py`.

- Ten requested clip-local times: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9` seconds.
- OpenCV `VideoCapture`; `fps = cap.get(cv2.CAP_PROP_FPS) or 30.0`.
- `total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))`.
- For each integer second `t`, seek with `CAP_PROP_POS_FRAMES` to
  `min(int(t * fps), max(0, total_frames - 1))`, then call `read()`.
- This is nominal-FPS frame-index selection. It does not select by presentation
  timestamp, validate the achieved timestamp, resample the whole video, or provide
  a verified variable-frame-rate policy. The requested times are not evidence of
  achieved decoder timestamps.
- Annotation `start`/`end` are event labels; they do not shift the ten-frame window.

## Pixels, layout and normalization

Producer code: `vision/extract_frames.py`, `vision/dataset.py:AVEVisual`.

1. Convert decoded OpenCV BGR to RGB using `cv2.COLOR_BGR2RGB`.
2. If necessary, resize directly to width 224, height 224 using
   `cv2.resize(..., interpolation=cv2.INTER_LINEAR)`.
   There is no crop, letterbox, aspect-ratio preservation or resize-to-256 step.
3. Cache `(10,224,224,3)` RGB `uint8` frames.
4. Permute to `(10,3,224,224)`, convert to float32, divide by `255.0`.
5. Per RGB channel, apply `(x - mean) / std`, with
   `mean = [0.485, 0.456, 0.406]` and `std = [0.229, 0.224, 0.225]`.
6. Export constructs `AVEVisual(split)` with `augment=False`, including TRAIN.
   No random flip, brightness scaling or temporal cutout is applied.

The dataset prefers an existing packed `{split}_X.npy`; otherwise it reads each
`frames_1fps/{clip_id}.npy`. Frame extraction skips existing nonempty cache files.
Consequently, current extraction code alone cannot prove how historical cached
pixels were created.

## Architecture, head and export

Producer code: `vision/model.py:VisionModel`, `vision/evaluate.py:load_model`
and `predict`, `vision/export_embeddings.py`.

- TorchVision ResNet-18; training starts with
  `ResNet18_Weights.DEFAULT` when pretrained is enabled.
- Replace ResNet `fc` with `nn.Identity()`. The encoder retains global average
  pooling and returns 512 values per frame. The 512-to-512 projection is Identity.
- Process frames independently by flattening batch/time; reshape to `(B,10,512)`.
  There is no temporal recurrent/attention layer or embedding L2 normalization.
- Separate trained classifier: `Dropout(0.3) -> Linear(512,29)`.
  Output classes are 28 alphabetically sorted AVE events, then `background` at 28.
  Fusion features are the pre-classifier embeddings, not logits/probabilities.
- Loader constructs with `pretrained=False` and loads `checkpoint['model']`;
  `checkpoint['config']` supplies backbone and embedding dimension.
- `predict` uses `model.eval()` and `torch.no_grad()`; no explicit export autocast
  is present. It collects float32 embeddings and softmax probabilities on CPU;
  export converts them to NumPy float16. Stored embedding shape is `(N,10,512)`.
- `SOURCE.txt` declares 10 epochs, cosine schedule, learning rate `3e-4`, batch 8,
  square-root class balancing. Notebook output records best epoch 9.
  The downloaded checkpoint confirms backbone, embedding dimension and epoch 9;
  it does not store all these training hyperparameters or the dropout probability.
- Rows join through `{split}_ids.csv` using `sample_id`, not `clip_id` alone.

## Missing and repeated frames

- Unopenable video: save ten black frames.
- Individual failed read: repeat the last valid resized RGB frame; until the
  first success, that frame is black.
- Short clips: clamp late requested indices to the final frame index; repeated
  successful reads are retained. No deduplication or interpolation occurs.
- Missing per-clip cache: dataset supplies ten black frames. Packed caches can
  also contain zeros for missing frame files.
- Export includes no decoder-validity or achieved-timestamp sidecar.

`src/utils/video.py:sample_10_frames` has a different failure policy (black on
each failed read). It is not the audited producer extraction path. The strict
fusion raw boundary rejects invalid decoded bins; do not silently treat legacy
substitutions as ten successfully decoded frames.

## Runtime evidence and hashes

| Item | Available evidence |
| --- | --- |
| Python | Vision notebook kernel metadata says `3.14.7`; export subprocess version is not independently recorded. |
| PyTorch | Notebook setup output says `2.14.1+xpu`, Intel Arc 140V. This is evidence about the notebook session, not a pinned export subprocess manifest. |
| TorchVision | Actual export version not recorded in available evidence. |
| OpenCV / decoder | Actual version, backend and codec build not recorded. Needed to reproduce pixel decoding/resizing. |
| Requirements | `requirements-video.txt` uses lower bounds; it is not an exact environment lock. |
| Trained `best.pt` SHA-256 | `6f1c1fc868efbd397d2ee1ca20c37800bd483911de6ef0727ac0f12871b8b2ec`; downloaded exact supplied file, 44,839,691 bytes. |
| Original export archive SHA-256 | Unavailable: no original archive found in inspected root/champion folders; individual export files were downloaded and hashed instead. |
| Actual local `train_emb.npy` SHA-256 | `bb1f0373c1820ae932e0a9e5732bd47ae51c641ae86271207a929b044ca7575d` |

The notebook export output lists embeddings/IDs, while current export source also
writes probabilities and class names. Do not infer the exact producer source
revision or environment solely from this notebook. Current local Python/package
versions are not evidence of the original export environment.

## TRAIN references and executable interfaces

The accompanying local handoff package contains three exact `(10,512)` float16
rows copied from the final TRAIN export, their original row indices/sample IDs,
source hashes, and this source configuration. They were not recomputed. The
selection is the first TRAIN row from each of the first three distinct categories;
no numeric feature values or held-out rows determine selection. Raw clips are not
included: none are local, and permission to redistribute particular clips is not
documented here. The manifest makes that limitation explicit.

There is no `VideoPredictor` class in this checkout. Existing executable paths:

- Producer export: `python -m vision.export_embeddings --run-id resnet18_v1`.
  Requires the trained checkpoint and original frame/metadata caches; exports all
  configured splits, so do not use it as a TRAIN-only parity command.
- Model interface: `vision/model.py:VisionModel.embed(x)` for normalized
  `(B,10,3,224,224)` tensors.
- Integration interface: `fusion/encoder_interfaces.py:load_encoder(EncoderSpec)`
  returns `FrozenEncoder`; `FrozenEncoder.encode(decoded)` accepts the strict
  decoded-media object and returns CPU float32 `(B,10,512)` embeddings. It requires
  a SHA-bound trained checkpoint, exact saved configuration and pinned runtime
  versions. It does not decode MP4 files. Its real-checkpoint parity is unverified.
- `fusion/end_to_end.py:TrainParityFusion` compares reconstructed embeddings to
  the original TRAIN exports with exact equality after the float16 roundtrip,
  before running frozen fusion. CPU/XPU equality has not been established.

`src/models/vision.py:VisionBranch` is an annotation-lookup demo with heuristic
fallback features, not trained ResNet inference.

Video parity remains **BLOCKED** pending:

1. Permitted original TRAIN clips for `c---zaDCTaE__08`, `0I_WwkebbeM__17`,
   `7KSXUptBBYQ__04` (or other explicitly matched TRAIN rows), with source hashes
   and decoder timing/validity evidence. Matching final embeddings/IDs are present.
2. Exact export-process Python/PyTorch/TorchVision/NumPy/OpenCV versions and
   decoder backend/codec details. Notebook runtime hints do not replace this.
3. Producer source/cache revision and confirmation binding the supplied checkpoint
   to the final exported features, followed by the unchanged exact TRAIN parity check.

Original export archive bytes are additionally required to report an original
archive SHA-256. A newly created reference-package checksum cannot substitute for
that archive checksum. The earlier reference package is an immutable snapshot of
the pre-acquisition audit; use this updated handoff and Drive verification record
for current artifact availability.

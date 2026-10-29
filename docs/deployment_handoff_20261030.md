# Final three-head private deployment handoff — 30 October 2026

This is a private experimental deployment candidate, prepared for review only.
No commit, push, PR creation, merge or deployment is authorized by this document.
Use [the latest diagnosis](prediction_diagnosis_20261030.md) as the scientific
baseline and [the Linux hosting runbook](production_hosting.md) for host setup.
Earlier reports and review-package hashes describe their historical candidates;
they do not approve this final source revision or its scientific release.

## Required source and application

Base main: `8a5e513dfbf4acd1c3adbf2e4a7c18f7236e840d` (merged PR #32).
The final PR must contain the combined event-evidence, trained Head 2, responsive
three-head UI, one-shot capture/finalization, physical-microphone diagnostics,
tests, training code/protocol and this documentation. The older 17-file UI and
28-file Head 2 allowlists are superseded by the final reviewed inventory.

The final commit does not exist yet. After explicit approval, record
`git rev-parse HEAD` of the resulting PR commit and its source manifest, then
use that exact revision for deployment. A base SHA, mutable branch/tag or earlier
image alone is insufficient. Deploy from a clean checkout of that commit;
never transfer the development workspace wholesale. Existing main already holds
the authenticated broker/worker, signed jobs, ownership/cancellation, decoder,
runtime locks, Dockerfile and `deployment/production/` service/preflight tools.
These inherited files are required even though they are absent from the PR diff.

Build the image from that clean checkout and record its immutable `sha256:` ID.
The default Docker CMD starts the ordinary gated application, **not** the private
three-head product. Launch the private product via `deployment.run_product_demo`
or the reviewed systemd unit, preserving its separate offline worker.

## Private artifact transfer: all three heads

Supply the 14 E3-profile files in `deployment/artifact_catalog.json`, plus
`Head2/weights.pt` and `Head2/metadata.json`. The base catalog SHA-256 remains
`a836d05cd9071282965578c964d3d555405ac02a2f631a9beb2539381bc95418`.
The Head 2 catalog SHA-256 is
`91ef1b5211e7dd50fb7e0353b7584339d32875d28a86f5e0953dc9f7e06f87d1`.
Every file's hash/size is in those checked-in catalogs. Critical weights:

| Relative file | SHA-256 |
| --- | --- |
| audio/encoder.pt | 8d1b234032a9ccff353612dc6c20982346dc2968b205b79d97303eb5e77bfb34 |
| audio/weights.pt | d4a0f78403caafde2c3c39d961c9819fbae8caebaaef4b824034c678da860692 |
| video/weights.pt | 6f1c1fc868efbd397d2ee1ca20c37800bd483911de6ef0727ac0f12871b8b2ec |
| E3/best.pt | e7df0de86d873005728c142c3bd31729df803f4cc805110d82e375fb5d695925 |
| Head2/weights.pt (703,161 bytes) | f957f83cf98f0f11e51217926f4519f93dcf4f04b43fa789e797e822a0dae8a7 |
| Head2/metadata.json (1,434 bytes) | ed0e9f4ff76843af9f4471ae9f6d96bccfe7ccbb4ca661f02fb8ab67fc6c71d4 |

Local originals: `_local/phase8_stage2c/models/{audio,video,E3}` and
`_local/temporal_presence/run-gru-v1/{weights.pt,metadata.json}`. The existing
`_local/temporal_presence/product-models-host` has the combined runtime layout.
Transfer only these 16 cataloged files to the authorized teammate through an
owner-approved encrypted channel, such as SSH/SFTP to a host whose fingerprint
has been verified separately. Keep the receiving staging directory private.
Do not include datasets, embeddings, TRAIN reference arrays or recordings. No
transfer or channel/recipient approval is claimed here. A2 is optional and is
not used by this application. Never publish checkpoints in Git, PR attachments,
container layers or a public model repository.

The existing private/pinned-HF provisioner covers the **base catalog only**;
it does not fetch Head 2. Transfer Head 2 separately via the approved channel.
Do not assume `--include-a2` includes Head 2. On the authorized Linux host,
as the selected UID-1000 service account, from the reviewed checkout:

```bash
python -B -m deployment.provision_artifacts \
  --destination /srv/avalidate/models --source-root /srv/avalidate/incoming
python -B -m deployment.temporal_presence \
  --source /srv/avalidate/incoming/Head2 --destination /srv/avalidate/models/Head2
python -B -m deployment.provision_artifacts \
  --destination /srv/avalidate/models --verify-only
python -B -c "from deployment.temporal_presence import verify; verify('/srv/avalidate/models/Head2'); print('HEAD2_VERIFIED')"
```

Use the pinned deployment virtualenv (`/opt/avalidate/venv/bin/python`) for
these commands. Apply the runbook's UID 1000/group 0 and directory 0750/file
0440 permissions to the entire verified models tree **including Head2**.
The supervisor seeds a new private volume and verifies/copies Head 2 if present.
Head 2 remains optional in generic product code; omission is unacceptable for
this three-head handoff. The existing Linux preflight verifies base artifacts
and sandbox permissions; the explicit Head 2 verification above and three-head
smoke below are also mandatory. Do not treat base-only preflight as Head 2 readiness.

## Exact host, launch and initial access

Supported profile: **Linux amd64/x86-64, cgroup v2, rootful Docker without
userns remapping, UID-1000 host broker, 4 vCPU/8 GiB RAM/20 GiB free SSD**.
Use pinned CPython 3.11.17 and the checked-in deployment hash locks. ARM,
rootless/remapped Docker and the current single-container HF Space have not
been qualified for this architecture. Model-worker restrictions remain:
network none; read-only root/models; UID/GID 1000; no capabilities;
no-new-privileges; 2 CPUs; 4 GiB RAM/no additional swap; 128 PIDs;
512 MiB temporary tmpfs. The trusted supervisor alone gets Docker access.

Follow `production_hosting.md` for installation, permissions, private config,
kernel preflight and systemd verification. Bind only to loopback. Before any
owner-authorized startup, set the actual immutable image ID in the private
service environment. A direct equivalent, after passing preflight, is:

```bash
/opt/avalidate/venv/bin/python -B deployment/production/preflight.py \
  --models /srv/avalidate/models --private-dir /var/lib/avalidate/private \
  --image sha256:ACTUAL_REVIEWED_IMAGE_ID
/opt/avalidate/venv/bin/python -B -m deployment.run_product_demo \
  --models /srv/avalidate/models --private-dir /var/lib/avalidate/private \
  --port 7862 --image sha256:ACTUAL_REVIEWED_IMAGE_ID --no-build
```

`ACTUAL_REVIEWED_IMAGE_ID` is an operator placeholder, not an executable value.
Initial remote access is **SSH only**:
`ssh -N -L 7862:127.0.0.1:7862 OPERATOR@VERIFIED_HOST`, then
`http://127.0.0.1:7862`. Expose no application/Docker ports. Preserve Host/Origin
checks and the Gradio login. Supply login secrets in the private server config
before first start, or let the supervisor generate `private/login.json`.
Existing credentials take precedence; never print/copy them into the PR.
Keep login/signing keys mode 0600 in UID-1000/mode-0700 private state.

## Required target acceptance

Browser-test prerequisites are installed Google Chrome and Node.js with
`playwright@1.64.1` at `_local/premium_ui/browser/node_modules/playwright` (the
existing harness's explicit dependency path). In an authorized test environment,
`npm install --prefix _local/premium_ui/browser --no-save --package-lock=false playwright@1.64.1`
provisions that local-only dependency. Python checks used pytest 9.1.1 in addition
to the pinned deployment dependencies. Keep test tooling outside the production
image/environment. The browser harness also needs its two authorized private
TRAIN fixture paths as documented in the test source; never commit those files.
Private Head 2 runtime tests need their documented cached TRAIN example and
skip explicitly if it is absent. Local test success is not target acceptance.

1. Confirm exact source commit/image, all 16 private hashes and sandbox checks.
   Verify worker health `READY`; login-page liveness alone is insufficient.
2. Through authenticated SSH access, upload a provenance-verified original TRAIN
   clip and a prespecified cross-class audio swap. Inspect **real** Head 1
   `(10,29)` evidence, `H2-AVE-presence-v1` ten values/threshold 0.25 and E3's
   raw score/threshold `0.009579099009671126`. Record outputs, not invented
   expected scores. Codec/runtime changes may change exact numerical results.
3. Run `tests/integration_product_demo.cjs` with its `AVAL_PRODUCT_TEST_URL`,
   `AVAL_PRODUCT_PRIVATE_DIR`, `AVAL_PRODUCT_TEST_OUTPUT` and
   `AVAL_PRODUCT_REQUIRE_HEAD2=1` settings after privately provisioning its
   authorized media and Playwright dependency. It uses fake devices only.
   Verify ownership denial, Cancel/Reset/Stop, no stale result and cleanup.
4. Run focused Python/JavaScript tests; test anonymous/private-file/foreign-Host
   refusal, revocation and stop/restart cleanup. Check both job spools contain
   no retained media or active grants. No physical-device automation is permitted.
5. Physical live demonstration is manual, with fresh consent. Show the actual
   full-frame source, audible input and diagnostics; press Reset afterward.
   Synthetic-device success does not resolve the reported physical-speaking case.

Local demo media/results and the professor guide are intentionally ignored in
`_local/professor_demo/`. They may be privately transferred if authorized;
redistribution rights have not been verified. TRAIN demo selection is not an
independent accuracy evaluation. No TEST media was selected or used for tuning.

## Scientific limitations and release boundary

- Exact 29-class mapping is verified, including Church bell 8, Clock 9,
  Flute 12, Goat 14, Male speech 17 and background 28.
- E3 is original-versus-audio-substituted pairing evidence. Score >=
  **0.009579099009671126 means MATCH**. Scores are uncalibrated and do not
  establish physical causation or semantic correctness. Validation false accepts
  were **51/402 (12.69%)**, on validation reused for selection; do not retune.
- Frozen Head 2's source-disjoint cached-embedding TEST macro-F1 is **0.706476**,
  accuracy 0.802284 versus always-presence 0.820305. Confusion matrix
  `[[455,253],[526,2706]]`; background recall 0.642655, presence recall 0.837252.
  Matched boundary MAE 1.711538 s; 64 missed/122 extra boundaries. Large
  TRAIN/held-out gap; no swapped/live temporal ground truth. See
  [Head 2 report](temporal_presence.md).
- Historical embedding parity is unresolved. Browser JPEG/physical processing
  is not numerically equivalent to the producer decoder. The diagnosis's
  codec-only bell control changed Head 2 presence from 2/10 to 6/10; it is not
  an exact physical-browser replay or a basis for choosing preprocessing.
- The synthetic bell diagnosis is complete at its documented scope; do not
  reopen class mapping, RGB, frame order or normalization. Physical speaking
  Case B remains unresolved without retained source/exact model-input evidence.
- Independent scientific/public release approval is absent. Public gates remain
  closed. The private product demonstrates experimental outputs only. No broad
  audit, retraining, threshold tuning or general model redesign is part of this PR.

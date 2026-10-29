# Private AValidate product demonstration

GC2 only. Authenticated upload and live prediction work through the original BEATs + LoRA / ResNet-18 / frozen E3 pipeline. This is a private demonstration, not independent scientific approval or public deployment. E3 threshold remains **0.009579099009671126**; ordinary public gates remain `MODEL_NOT_READY`.

## Launch and login

Requirements: the existing Python environment, running Docker with Linux containers, and the original private catalog-verified model directory. Run from the repository root. The following instance is already running at **http://127.0.0.1:7862**; do not start a second supervisor against its locked private directory.

```powershell
$py = '.\_local\phase8_stage1\venv\Scripts\python.exe'
& $py -B -m deployment.run_product_demo --models '.\_local\phase8_stage2c\models' --private-dir '.\_local\product_demo\private' --port 7862
```

The command builds the image, verifies/provisions the original E3 catalog into a temporary private volume, starts the localhost broker and starts the no-network worker. Use `--no-build` only when reusing the already verified current image. No HF token or physical device is needed to start.

The supervisor always configures `AVAL_ACCESS_MODE=gradio_auth`. On first start it uses optional private `AVAL_GRADIO_USERNAME`/`AVAL_GRADIO_PASSWORD` environment values, or generates username `avalidate` and a random password. It saves them only in `_local/product_demo/private/login.json`; the operator can read that local file to log in. Existing credentials are reused on restart. The password/key are never printed by the launcher or served by the UI. Keep the private directory and all `_local` files out of Git.

For operator-chosen credentials, set these **before first start with a fresh private directory**:

```powershell
$credential = Get-Credential -Message 'GC2 private demonstration login'
$env:AVAL_GRADIO_USERNAME = $credential.UserName
$env:AVAL_GRADIO_PASSWORD = $credential.GetNetworkCredential().Password
```

The broker sets its own private temp/model-placeholder paths. Only the isolated worker receives verified model bytes. Stop with Ctrl+C in the supervisor terminal, or from another terminal:

```powershell
& '.\_local\phase8_stage1\venv\Scripts\python.exe' -B -m deployment.run_product_demo --private-dir '.\_local\product_demo\private' --stop
```

Stop revokes jobs, clears media and removes the temporary model volume. Credentials remain private for restart. The signing key is separate from login credentials and is mounted read-only into the worker.

## Product behavior and executed acceptance

Select **Verify uploaded video**, choose a complete supported 10-second H.264/AAC MP4/MOV (≤50 MiB), wait for Verify to enable, then click it. Results show classification, raw uncalibrated E3 score, frozen threshold and interpretation. Clear the current file to select another. Cancel/reset revokes the pending signed upload job and suppresses stale results.

Select **Verify live capture** for completed-window signed-ticket replay. Explicit consent is required for each capture, including after Reset. Stop/Reset/logout/expiry invalidate authorization. Automated acceptance used Chrome fake devices exclusively; no physical recording is authorized by this implementation or document.

Latest actual browser results (not benchmark accuracy):

| Input | Genuine raw E3 score | Classification | Model processing |
| --- | ---: | --- | ---: |
| Authorized original TRAIN video | 0.9999999996978804 | Original-like pairing | 2.642 s |
| Prespecified TRAIN cyclic audio substitution | 0.007293270978074229 | Audio-substituted-like pairing | 2.596 s |
| Completed fake-device live window | 0.004391256462102586 | Audio-substituted-like pairing | 2.133 s |

Invalid media returned `GENERAL_DECODE_FAILED`, null predictions and no score. The combined browser test passed foreign-login upload/live denial, rightful-owner consumption, upload cancellation without stale display, live Reset revocation, renewed consent, live Stop, media deletion, private-file route denial and foreign-Host denial. No browser page errors occurred. Fake-device content/timestamps vary; its score is not a natural-media validity claim.

Final tests: **151 focused**, **1,141 Python passed + 1,346 subtests**, two CUDA skips and one upstream warning; **18 JavaScript passed**; **149 Docker passed**, plus image/source checks and actual networked-worker refusal. Atomic-write tests verify both transient-reader recovery and preservation of the prior signed value on permanent replacement failure. No test was weakened.

Reproduce with `pytest tests`, `node --test tests/js/*.js`, and `node tests/integration_product_demo.cjs` against the separately launched demo. The browser harness uses only fake devices and expects the existing authorized TRAIN source and locally prepared cyclic-substitution fixture; private fixtures/dependencies are not Git deliverables. Exact logs, screenshots, resource measurements and sanitized results are under `_local/product_demo/`; `product-handoff.json` records current source/evidence hashes. Earlier failed browser attempts are not counted as passes.

## Hosting: verified requirements and remaining step

Current local image: `sha256:f32f301c533b0f234f3e866643ab6b2be2d6c708e5e969363596a1805fd206bc`, about 2.67 GB. Measured startup including private provisioning: 9.39 s; worker initialization: 3.26 s; peak worker RSS about 1.11 GiB and cgroup peak about 1.39 GiB in these fixtures. These are local measurements, not a hosted SLA. After acceptance there were zero pending grants and zero retained capture/upload-media files in either job spool.

The implemented portable architecture is an operator-managed Docker host with a loopback authenticated broker and separate signed upload/live spools. Worker requirements are enforced by kernel checks: nonroot UID, network none, read-only root/models, zero capabilities, no-new-privileges, ≤4 GiB memory/no extra swap, ≤2 CPUs, ≤128 PIDs and bounded temporary storage (tested `/tmp` 512 MiB). The host also needs broker/OS/build headroom and private disk for image/model volume. The Docker socket stays with the trusted host supervisor, never the browser or model worker. One model instance handles both queues; startup/job watchdogs remain bounded.

For a private remotely hosted demonstration, an owner can provision a trusted Linux Docker VM, transfer the verified private artifacts and exact reviewed source, install the pinned dependencies, and run the same supervisor. Access it through an authenticated SSH tunnel preserving the localhost Host/Origin checks, for example `ssh -L 7862:127.0.0.1:7862 operator@HOST`. No VM has been provisioned or deployed by this task.

The existing private HF Docker Space was read-only verified at `9eb201ef60d343ef486e521e98ef12b5c3ea38c2`, RUNNING on CPU Basic. Its Dockerfile still starts the ordinary gated `deployment.gradio_app`; it contains neither the new product broker nor worker. Its hardware capacity alone does not attest the mandatory isolated worker controls. No supported current-Space mechanism for enforcing that separate no-network/read-only/cgroup-limited worker has been verified, so genuine inference is **not ready to deploy there**. [HF Docker Spaces documentation](https://huggingface.co/docs/hub/spaces-sdks-docker) describes the hosted app container/port, not evidence that this candidate's worker sandbox is satisfied.

**Exact next engineering step:** the infrastructure owner must provide an independently enforceable worker sandbox (verified on the target host) and immutable private artifact provisioning, either through an explicitly supported HF mechanism or the implemented self-managed Docker-host architecture. Validate the target with the same tests before any deployment authorization. Public prediction release additionally requires independent scientific sign-off bound to this exact product revision, numerical/media/live scope and unchanged models; prior review-package hashes identify the earlier candidate and do not approve this new diff. This task changes no release gate and performs no commit, push, merge or deployment.

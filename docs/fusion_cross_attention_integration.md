# Phase 5 integration and Windows environment verification

Date: 2026-10-25, Asia/Singapore. **READY TO COMMIT.**
This follow-up supersedes the earlier environment and upstream readiness
snapshot; it does not change frozen research results or experiment evidence.

## Git dependency and scope

Branch: `feature/fusion-phase5-cross-attention`.
HEAD: `38d4171d1b9c6aefab476a87fa8fc144999191d9`.
After fetching origin, main is
`ced1c3a1abf41e4c2bc05e2eeff96530e185249a`, merge PR #16.
Phase 4 commit `76add86568daed6f7dbb31659f26c020b264962c` is an ancestor
of main. The Phase 4 dependency is now merged.

The committed HEAD and origin/main trees are identical. Their histories have
two merge bases, `467c756` and `76add86`. Consequently,
`git diff origin/main...HEAD` currently warns about multiple merge bases and
can misleadingly show the old Phase 4 additions. The endpoint check
`git diff --exit-code origin/main HEAD` passes. The current working difference
against main consists of six necessary shared-file modifications (+188/−43)
and the Phase 5 additions listed below. No Phase 4-only file belongs in this
staging inventory.

The intended index contains **23 Phase 5 text files: six modifications and
17 additions**. No raw embeddings, weights, checkpoints, local experiment
directories, credentials or teammate changes are included.

```text
A docs/fusion_cross_attention_ablations.md
A docs/fusion_cross_attention_comparison.md
A docs/fusion_cross_attention_foundation.md
A docs/fusion_cross_attention_integration.md
A docs/fusion_cross_attention_phase5_freeze.json
A docs/fusion_cross_attention_phase5_summary.md
A docs/fusion_cross_attention_training.md
A fusion/cross_attention_ablations.py
A fusion/cross_attention_model.py
A fusion/cross_attention_tracking.py
M fusion/embedding_data.py
M fusion/mlp_study.py
M fusion/mlp_tracking.py
A fusion/model_interface.py
A fusion/smoke_cross_attention.py
M fusion/smoke_mlp.py
A fusion/train_cross_attention.py
M fusion/train_mlp.py
M fusion/training_contracts.py
A tests/test_fusion_checkpoint_binding.py
A tests/test_fusion_cross_attention.py
A tests/test_fusion_cross_attention_ablations.py
A tests/test_fusion_cross_attention_training.py
```

`requirements-fusion-training.txt` remains untracked and excluded. Its purpose
is a local installer convenience for the experiment SDK and official-reference
adapter (`wandb==0.31.0`, `h5py==3.16.0`, plus the tracked MLP requirements).
It predates Phase 5 and introduces no necessary change to the validated model
or test dependency contracts. Its bytes remain unchanged:
`a01116f4399965d31d989d92ec8d14a07cd518b55208d42a9081efd91f2848ab`.

## Supported Windows diagnosis and current recovery result

The earlier error is confirmed as a block of **c10.dll itself**, rather than an
unidentified dependency. Code Integrity Operational event 3077, record 184858,
at 2026-10-25 21:47:13 Singapore time names the GC2 environment's
`torch/lib/c10.dll` and policy `VerifiedAndReputableDesktop`, GUID
`{0283ac0f-fff1-49ae-ada1-8a933130cad6}`.
Its status is `0xc0e90002`; requested/validated signing levels are 2/1.
Correlated event 3089 (records 184857 and 184859, activity
`{32e73675-5bf1-0013-422c-cf44f15bdd01}`) reports zero signatures and an
unknown publisher. Authenticode also reports the file as unsigned.

Defender reports `SmartAppControlState=On`. The registry value
`VerifiedAndReputablePolicyState=1` and DeviceGuard user-mode/code-integrity
enforcement status 2 agree. Microsoft identifies
`VerifiedAndReputableDesktop` as the Smart App Control enforcement policy:
[Smart App Control modes](https://learn.microsoft.com/en-us/windows/apps/develop/smart-app-control/test-your-app-with-smart-app-control).
Its [App Control diagnostic guide](https://learn.microsoft.com/en-us/windows/security/application-security/application-control/app-control-for-business/operations/appcontrol-debugging-and-troubleshooting)
describes correlation of 3077 blocks with 3089 signature events.
CiTool policy enumeration returned access denied, so unrelated policy inventory
was not established; no broader enterprise-policy claim is made.

The existing `.venv/Scripts/python.exe` now imports **PyTorch 2.14.1+cpu**
successfully under the active security policy. Its c10.dll is 1,113,600 bytes;
SHA-256
`5e9ee43835f118e1d4bed62ddb4406df947fce8c9127452a79c71a2a3b919484`
matches both the installed wheel RECORD and the flat hash in the blocked event.
The registered global Python 3.14 and workspace base Python 3.11 have no
PyTorch installation; no alternative environment was required.

No policy was disabled or changed, no permission or DLL-unblocking workaround
was used, and no binary/package was installed, replaced or reinstalled.
The available diagnostics do not establish why the same binary's later loading
was allowed. Current successful execution is verified; permanent future
allowability is not guaranteed. If the block recurs, use an approved environment
or an administrator-approved allowance for the verified distribution.

## Fresh verification and preservation

The required suite completed in this same environment:

```text
Ran 478 tests in 81.429s
OK (skipped=2)
```

That is **476 passes and two CUDA skips**, not a reused historical result.
Safe checkpoint verification reproduced every one of the 804 primary-validation
logits exactly for the initial model, A1, A2 and A3. Model/configuration binding,
parameter counts, selected epochs, selected optimizer steps, source archives,
output manifests and thresholds passed. Existing verifier scripts were executed
with their final report writes suppressed, preserving historical evidence.

All **280 protected historical files / 126,941,543 bytes**, all **92 existing
Phase 5 run files**, frozen source hashes and the initial run's 23-file inventory
remain unchanged. This preserves the four Phase 4 checkpoint bytes too.
Python syntax, whitespace, dependency and credential-pattern checks pass.

A2 remains the selected attention variant by primary-validation AUC;
E3 remains the best learned model by validation AUC. All scientific limitations
and boundaries in the [final research summary](fusion_cross_attention_phase5_summary.md)
remain in force. No real training, tuning, challenge/test inference, new Forge
run or artifact version was performed.

The [freeze JSON](fusion_cross_attention_phase5_freeze.json) remains byte-identical
(SHA-256 `4fcbd1537ab3e5574ee6d0d06d7d3a2787431741dd14cc6c161577b0a689f9e5`).
Its previous blocker and unmerged-main entries are historical freeze-session
facts, superseded for current integration readiness by this document.

## Commands to complete integration manually

The intended files have been staged and the index checked. No commit, push,
merge, rebase, branch switch or PR creation was performed. Run the following
blocks in order and stop if any command fails.

Review and commit the verified index:

```powershell
git diff --cached --check
git diff --cached --stat
git diff --cached --name-status
git commit -m "feat(fusion): add audited Phase 5 cross-attention study"
```

After committing, preserve the separately untracked requirements file before
synchronizing the branch. The stash command is scoped to that one file:

```powershell
git stash push --include-untracked -m "preserve local fusion training requirements" -- requirements-fusion-training.txt
git fetch --no-tags origin
git merge --no-edit origin/main
```

Only after a successful merge, restore the local file while retaining the stash
as a backup. If a conflict occurs, stop and resolve it without overwriting files.

```powershell
git stash apply 'stash@{0}'
git status --short
git diff --check origin/main...HEAD
git diff --name-status origin/main...HEAD
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

After synchronization, main is an ancestor of the branch and the three-dot PR
diff must contain only the intended Phase 5 inventory. The only expected
unstaged/untracked remainder is the preserved requirements file.

Push only after that review and successful verification:

```powershell
git push --set-upstream origin feature/fusion-phase5-cross-attention
```

GitHub CLI is not installed on this computer. After pushing, the browser route
requires no new local installation: open the
[main-to-Phase-5 comparison](https://github.com/NUS-AIS-GCs/Audio-Video-Confidence/compare/main...feature/fusion-phase5-cross-attention?expand=1),
confirm base `main` and compare `feature/fusion-phase5-cross-attention`, review
the file list, and create the PR using the title and body in the command below.
The equivalent CLI command is provided for an environment with an already
installed and authenticated official GitHub CLI:

```powershell
gh pr create --repo NUS-AIS-GCs/Audio-Video-Confidence --base main --head feature/fusion-phase5-cross-attention --title "Phase 5: audited cross-attention fusion and controlled ablations" --body "Adds architecture-bound bidirectional and directional cross-attention models, shared training integration and three controlled ablations. Freezes A2 as the selected attention variant while E3 MLP remains the best learned validation-AUC model. Validation: 478 tests (476 passes, two CUDA skips); all four attention checkpoints reproduce 804 primary-validation logits exactly. Historical artifacts are unchanged. Single-seed, validation-reuse and source-overlap limitations are documented."
```

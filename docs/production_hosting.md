# GC2 private Linux hosting — prepared, not deployed

For the final three-head candidate, also follow the
[30 October handoff](deployment_handoff_20261030.md): provision and verify
`Head2/weights.pt` and `Head2/metadata.json` separately. Base-only provisioning
or base-only preflight does not establish three-head readiness. Use the latest
reviewed source commit and image; historical measurements below are context.

Use the existing `deployment.run_product_demo` supervisor, localhost broker and signed upload/live spools. One isolated worker loads the original models and handles both queues. No new HTTP inference service, public reverse proxy, model change or release-gate activation is introduced. The target host must be confirmed by the owner before executing installation/start commands.

## Supported host profile

Provision a dedicated **Linux x86-64 host, cgroup v2, 4 vCPU, 8 GiB RAM and at least 20 GiB free SSD**. This is the selected supported host profile, not a measured theoretical minimum. Existing local measurements: image about 2.67 GB, startup/provisioning 9.39 s, worker initialization 3.26 s, cgroup peak about 1.39 GiB on limited fixtures. Budget additional broker/supervisor/OS/build headroom; no hosted SLA is claimed.

- Docker Engine must be rootful with **no userns remapping**. The image worker is UID/GID `1000:1000`; the host broker/supervisor must run as **UID 1000**. Rootless/remapped Docker is not silently interchangeable with these bind-mount permissions.
- The inference worker retains `--network none`, read-only root/model volume, nonroot UID, `--cap-drop ALL`, no-new-privileges, 4 GiB memory/no extra swap, two CPUs, 128 PIDs and 512 MiB `/tmp` tmpfs. The kernel preflight must pass on the actual target.
- The trusted host supervisor needs Docker-daemon access; Docker-group membership is effectively host-administrator authority. Never mount its socket into the broker or worker. Use a dedicated trusted account/host, not a shared untrusted Docker service.
- Persist source/image revision, catalog-verified private models, and private login/signing-key state. Do not back up/restore grants, capture/upload spools, temp files or `ready.json` as authorization. Encrypt disks/backups; back up login/key only under the operator's secret-handling policy.
- Bind the broker only to **127.0.0.1:7862**. Expose only allowlisted SSH (normally TCP 22); do not open 7860/7862/80/443 or publish Docker ports. Clients use `ssh -N -L 7862:127.0.0.1:7862 OPERATOR@HOST`, then http://127.0.0.1:7862. Keep SSH host-key verification enabled.
- Transport encryption/authentication is SSH; HTTP exists only on loopback. This selected architecture has no public TLS endpoint and needs no public certificate. A mandated HTTPS/private-VPN ingress is a separately reviewed scope; do not expose the broker through a public TLS proxy to bypass its Host/Origin restrictions. Gradio login remains required inside the tunnel.

## Host preparation after target confirmation

The owner supplies an existing UID-1000 trusted service account, SSH access, Docker/cgroup-v2 installation and an exact reviewed source checkout at `/opt/avalidate/app`. Do not change an unrelated user's UID or copy the entire Windows workspace. Transfer only reviewed Git source and the approved private models. Commands below run **on the confirmed Linux host**, not in the current Windows workspace.

```bash
getent passwd 1000                    # owner must confirm this exact account
id -u SERVICE_ACCOUNT               # must print 1000
sudo usermod -aG docker SERVICE_ACCOUNT
sudo install -d -o root -g root -m 0755 /opt/avalidate
sudo install -d -o 1000 -g 0 -m 0755 /opt/avalidate/venv
sudo install -d -o 1000 -g 0 -m 0750 /srv/avalidate
sudo install -d -o 1000 -g 0 -m 0700 /srv/avalidate/incoming
sudo install -d -o 1000 -g 0 -m 0700 /var/lib/avalidate/private
sudo install -d -o root -g root -m 0700 /etc/avalidate
```

Install **CPython 3.11.17** through the owner's approved provisioning method; do not replace the system Python. Create `/opt/avalidate/venv` for UID 1000. As that service account, from the reviewed source:

```bash
cd /opt/avalidate/app
/opt/avalidate/python/bin/python3.11 -m venv /opt/avalidate/venv
/opt/avalidate/venv/bin/python -m pip install --only-binary=:all: --require-hashes \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r deployment/requirements-runtime.lock -r requirements-deployment.txt
/opt/avalidate/venv/bin/python -m pip check
docker build -t avalidate:stage1-reviewed .
docker image inspect avalidate:stage1-reviewed --format '{{.Id}}'
```

Build/installation need controlled outbound HTTPS to approved package/image registries. Any private artifact download also runs before inference, outside the worker. Hash verification remains mandatory; do not remove version/hash pins to make installation pass. The independently recorded image ID must come from this actual build, not an old local tag/attestation.

## Private model provisioning and Linux permissions

Supply the **14 required E3-profile files** at their catalog-relative paths under `/srv/avalidate/incoming`: audio base/LoRA/head/card/normalization/class/environment, video weights/card/class/environment, and E3 checkpoint/metadata/integrity/operating point. Reuse `deployment/artifact_catalog.json`, SHA-256 `a836d05cd9071282965578c964d3d555405ac02a2f631a9beb2539381bc95418`. No TRAIN media, historical arrays or A2 execution is required for product startup. No weights or credentials enter Git/the image.

As UID 1000, provision a **fresh** destination:

```bash
/opt/avalidate/venv/bin/python -B -m deployment.provision_artifacts \
  --destination /srv/avalidate/models --source-root /srv/avalidate/incoming
```

Alternatively use the existing provisioner's `--repo PRIVATE_MODEL_REPO --revision IMMUTABLE_40_HEX_COMMIT` and supply its read-only `AVAL_ARTIFACT_READ_TOKEN` privately only during provisioning. An owner must supply that private source/commit/token; none is embedded here. Unset the token before starting the service. Existing incomplete/mismatched destinations fail closed, never overwrite in place.

The capability-dropped root seeder cannot override Linux DAC permissions. Grant read access through **group 0**, not world-read permissions. Apply only to this explicitly named, verified model directory:

```bash
test "$(readlink -f /srv/avalidate/models)" = /srv/avalidate/models || exit 1
sudo chown -R 1000:0 /srv/avalidate/models
sudo find /srv/avalidate/models -type d -exec chmod 0750 {} +
sudo find /srv/avalidate/models -type f -exec chmod 0440 {} +
```

The broker's private directory is UID 1000/mode 0700; its key/login are mode 0600. The worker reads those as the same UID. Do not solve access failures with 0777, extra capabilities, root inference or network-enabled workers. The supervisor seeds a new model volume and mounts it read-only; it removes this temporary volume on normal stop. Source models and login/key state persist.

## Configuration, preflight and startup

```bash
sudo install -o root -g root -m 0600 deployment/production/product.env.example /etc/avalidate/product.env
sudoedit /etc/avalidate/product.env
```

Set `AVAL_PRODUCT_IMAGE=sha256:ACTUAL_64_HEX_IMAGE_ID` to the immutable inspected build ID. Optionally enter `AVAL_GRADIO_USERNAME`/`AVAL_GRADIO_PASSWORD` privately **before first startup**; if omitted the existing supervisor generates a random login in `/var/lib/avalidate/private/login.json`. Existing login.json takes precedence. Do not put secrets in shell history, source, logs or this template. The service always uses `gradio_auth`; no numerical/public-readiness bypass exists.

As UID 1000, replace the placeholder below with that same reviewed image ID. This checks hashes, exact Linux ownership, root-seeder read access and the actual UID-1000 worker sandbox using **disposable non-secret** probe bytes. It neither loads models nor starts the product:

```bash
/opt/avalidate/venv/bin/python -B deployment/production/preflight.py \
  --models /srv/avalidate/models --private-dir /var/lib/avalidate/private \
  --image sha256:ACTUAL_64_HEX_IMAGE_ID
```

Only after preflight passes and the owner authorizes starting this private target:

```bash
sudo install -o root -g root -m 0644 deployment/production/avalidate-product.service /etc/systemd/system/avalidate-product.service
sudo systemd-analyze verify /etc/systemd/system/avalidate-product.service
sudo systemctl daemon-reload
sudo systemctl start avalidate-product
sudo systemctl status avalidate-product --no-pager
# Optional boot enablement only after target acceptance:
# sudo systemctl enable avalidate-product
```

`Type=simple` means systemd “active” alone does not attest readiness. Check the fresh `/var/lib/avalidate/private/ready.json`, worker health, exact image/limits and the authenticated UI through SSH. Execute authorized upload and fake-device acceptance on the target; synthetic fixtures are not natural-media validation. Never activate physical devices without fresh consent. Stop with `sudo systemctl stop avalidate-product`; inspect `cleanup.json` for model-volume removal.

The unit deliberately does not auto-restart after failure. A forced SIGKILL can leave a bounded Docker worker/volume outside systemd's process group. Before manual restart, inspect the exact worker name in private `ready.json` and Docker's command/mount/image metadata, stop only that GC2 worker and remove only its associated temporary volume after it stops. Never run global Docker prune or touch another project's containers. Source model/credential directories must be retained. Monitor disk and resource exhaustion; no availability SLA or automatic failover is claimed.

## HF decision and external dependencies

The existing private Docker Space was read-only checked through the owner's configured account as RUNNING/CPU Basic at `9eb201ef60d343ef486e521e98ef12b5c3ea38c2`; its default command is the gated `deployment.gradio_app`. An app container/port and sufficient hardware do not establish the required separate offline worker controls. No supported current-Space mechanism satisfying that sandbox has been verified. Do not deploy genuine inference there by suppressing `require_research_sandbox`. Sources: [HF Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker), [Docker UID mapping](https://docs.docker.com/engine/security/rootless/uid-gid-mapping/), [bind mounts](https://docs.docker.com/engine/storage/bind-mounts/).

Still required from the owner: confirmed host/provider/region and administrator/SSH credentials/host fingerprint; suitable UID-1000 account and trusted Docker administration; private disk/backup policy; verified artifact files or private immutable model repository/read token; reviewed source commit/image ID; operator login/secret custody; firewall/SSH policy; and authorization to run target acceptance/start the service. No target has been contacted or deployed by this preparation. Public upload/live release additionally needs the independent scope-specific scientific/release decision and a separately reviewed code-owned gate change. Until then both public gates stay closed.

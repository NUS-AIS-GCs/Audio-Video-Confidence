"""Target preflight inside the existing worker image; no model execution or HTTP."""
import json
from pathlib import Path
import os

from deployment.general_decode import require_research_sandbox
from deployment.provision_artifacts import verify_artifacts
from deployment.release_gate import unapproved_prediction


def main():
    proof=require_research_sandbox()
    if os.getuid()!=1000:raise RuntimeError('WORKER_UID_MISMATCH')
    report=verify_artifacts(Path('/source-models'))
    if not report['ready']:raise RuntimeError('MODEL_CATALOG_NOT_READY')
    # The host creates disposable non-secret bytes with the same modes as spools/key.
    root=Path('/permission-probe')
    if (root/'key').read_bytes()!=b'GC2_PERMISSION_PROBE_NOT_A_SECRET':
        raise RuntimeError('PRIVATE_KEY_PERMISSION_MISMATCH')
    target=root/'spool'/'write-probe'
    try:
        with target.open('xb') as stream:stream.write(b'probe')
    finally:target.unlink(missing_ok=True)
    if any(unapproved_prediction(scope)['inference_enabled'] for scope in ('upload','live')):
        raise RuntimeError('PUBLIC_GATE_CHANGED')
    print(json.dumps({'sandbox':proof,'catalog_ready':True,'uid1000_private_io':True,
                      'public_gates_closed':True},sort_keys=True))


if __name__=='__main__':main()

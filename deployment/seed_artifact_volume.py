"""Authenticate and copy a TRAIN snapshot into a fresh, temporary Docker volume."""
import argparse
import json
from pathlib import Path
import shutil
import time

from deployment.train_manifest import load_manifest, pinned_child, verify_file
from fusion.contracts import ContractError


def copy_verified_files(files, source_root, cache_root):
    source_root,cache_root=Path(source_root).resolve(),Path(cache_root).resolve()
    if source_root==cache_root or source_root.is_relative_to(cache_root) or cache_root.is_relative_to(source_root):
        raise ContractError('source and snapshot roots must be separate')
    total=0
    for relative,digest in files.items():
        source=pinned_child(source_root,relative);destination=pinned_child(cache_root,relative)
        verify_file(source,digest)
        destination.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as incoming,destination.open('xb') as outgoing:
            shutil.copyfileobj(incoming,outgoing)
        verify_file(destination,digest)
        destination.chmod(0o444);total+=destination.stat().st_size
    return total


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest_sha256')
    parser.add_argument('--source-root',type=Path,default=Path('/source-models'))
    parser.add_argument('--inputs-root',type=Path,default=Path('/inputs'))
    parser.add_argument('--cache-root',type=Path,default=Path('/cache'))
    parser.add_argument('--output',type=Path,default=Path('/results/seed.json'))
    args=parser.parse_args(argv);started=time.perf_counter()
    manifest_path=args.inputs_root/'manifest.json'
    manifest=load_manifest(manifest_path,args.manifest_sha256,args.source_root,args.inputs_root)
    total=copy_verified_files(manifest['model_files'],args.source_root,args.cache_root)
    verify_file(manifest_path,args.manifest_sha256)
    record={'files':len(manifest['model_files']),'bytes':total,'source_and_snapshot_hashes_verified':True,
            'elapsed_seconds':time.perf_counter()-started,'manifest_sha256':args.manifest_sha256}
    with args.output.open('x',encoding='utf-8') as stream:stream.write(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record));return 0


if __name__=='__main__':raise SystemExit(main())

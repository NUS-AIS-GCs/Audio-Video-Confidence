"""Hosting permission regressions: catch Linux DAC failures hidden by Windows mounts."""
from types import SimpleNamespace

import pytest

from deployment.production import preflight


def test_uid_alignment_rejects_a_worker_inaccessible_private_key():
    preflight.require_identity('linux',1000,['name=seccomp,profile=builtin'])
    with pytest.raises(RuntimeError):preflight.require_identity('linux',1001,[])
    with pytest.raises(RuntimeError):preflight.require_identity('win32',1000,[])


@pytest.mark.parametrize('option',['name=rootless','name=userns'])
def test_uid_remapping_is_not_silently_accepted(option):
    with pytest.raises(RuntimeError):preflight.require_identity('linux',1000,[option])


def test_private_spool_does_not_become_world_readable_to_fix_permissions():
    preflight.require_private(SimpleNamespace(st_uid=1000,st_mode=0o40700))
    with pytest.raises(RuntimeError):preflight.require_private(SimpleNamespace(st_uid=1000,st_mode=0o40755))
    with pytest.raises(RuntimeError):preflight.require_private(SimpleNamespace(st_uid=0,st_mode=0o40700))


def test_root_seeder_group_read_does_not_expose_models_to_other_users():
    preflight.require_model_permissions(SimpleNamespace(st_uid=1000,st_gid=0,st_mode=0o100440),False)
    preflight.require_model_permissions(SimpleNamespace(st_uid=1000,st_gid=0,st_mode=0o40750),True)
    for group,mode in [(1000,0o100440),(0,0o100444),(0,0o100600)]:
        with pytest.raises(RuntimeError):
            preflight.require_model_permissions(SimpleNamespace(st_uid=1000,st_gid=group,st_mode=mode),False)

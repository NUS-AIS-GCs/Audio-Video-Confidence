"""Private, signed local research spool. This is NOT a scientific release gate.

Only the local authenticated capture process creates grants. A trusted operator
mounts the key and private spool into an offline inference container. No HTTP
inference endpoint, client paths, archive extraction, cookies or weights here.
"""
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import time

from deployment.live.buffer import LiveBufferError

TTL = 300
AUTH_LEASE = 5  # broker loss cannot leave an offline authorization valid for five minutes
MAX_CAPTURE = 8 * 1024**2
MAX_RESULT = 128 * 1024


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def digest(data):
    return hashlib.sha256(data).hexdigest()


class ResearchStore:
    def __init__(self, root, key_file, *, clock=time.time):
        if Path(root).is_symlink():raise LiveBufferError('LIVE_RESEARCH_PRIVATE_SPOOL_REQUIRED')
        self.root = Path(root).resolve()
        self.clock = clock
        key_path = Path(key_file)
        if key_path.is_symlink() or not key_path.is_file() or key_path.stat().st_size != 32:
            raise LiveBufferError('LIVE_RESEARCH_KEY_REQUIRED')
        self._key = key_path.read_bytes()
        if not self.root.is_dir() or self.root.is_symlink():
            raise LiveBufferError('LIVE_RESEARCH_PRIVATE_SPOOL_REQUIRED')

    def path(self, ticket, suffix):
        if type(ticket) is not str or not re.fullmatch('[a-f0-9]{64}',ticket):
            raise LiveBufferError('LIVE_RESEARCH_NOT_FOUND')
        path = self.root/(ticket+suffix)
        if path.is_symlink():
            raise LiveBufferError('LIVE_RESEARCH_NOT_FOUND')
        return path

    def _sign(self, value):
        raw = canonical(value)
        return canonical({'body':value,'mac':hmac.new(self._key,raw,'sha256').hexdigest()})

    def read(self, ticket, suffix, maximum):
        try:
            path = self.path(ticket,suffix)
            if path.stat().st_size > maximum:
                raise ValueError
            with path.open('rb') as stream:raw=stream.read(maximum+1)
            if len(raw)>maximum:raise ValueError
            def unique(pairs):
                out={}
                for k,v in pairs:
                    if k in out:raise ValueError
                    out[k]=v
                return out
            envelope=json.loads(raw,object_pairs_hook=unique)
            if set(envelope)!={'body','mac'} or not hmac.compare_digest(
                    envelope['mac'],hmac.new(self._key,canonical(envelope['body']),'sha256').hexdigest()):
                raise ValueError
            return envelope['body']
        except Exception:
            raise LiveBufferError('LIVE_RESEARCH_NOT_FOUND') from None

    def write(self, ticket, suffix, value, *, replace=False):
        raw=self._sign(value)
        limit=MAX_CAPTURE if suffix=='.capture.json' else MAX_RESULT
        if len(raw)>limit:raise LiveBufferError('LIVE_RESEARCH_CAPTURE_TOO_LARGE')
        target=self.path(ticket,suffix)
        if not replace and target.exists():raise FileExistsError
        # The capture registry serializes issuance; the offline CLI has an
        # exclusive process lock. Publish complete files so polling cannot read
        # a partially written result and mistake it for revocation/tampering.
        temporary=self.path(ticket,'.tmp-'+secrets.token_hex(8))
        try:
            with temporary.open('xb') as stream:stream.write(raw)
            os.chmod(temporary,0o600);os.replace(temporary,target)
        finally:temporary.unlink(missing_ok=True)

    def issue(self, owner, capture):
        ticket=secrets.token_hex(32);cancel=secrets.token_hex(32);now=self.clock()
        capture={**capture,'ticket':ticket,'owner':owner,'created':now,'expires':now+TTL}
        grant={'ticket':ticket,'owner':owner,'created':now,'expires':now+TTL,'active':True,
               'lease_until':now+AUTH_LEASE,
               'capture_sha256':digest(canonical(capture)),'cancel_sha256':digest(cancel.encode())}
        try:
            self.write(ticket,'.capture.json',capture)
            self.write(ticket,'.grant.json',grant)
        except Exception:
            self.delete_capture(ticket);raise
        return ticket,cancel

    def check(self, ticket, owner=None):
        grant=self.read(ticket,'.grant.json',MAX_RESULT)
        if (grant.get('ticket')!=ticket or grant.get('active') is not True
                or type(grant.get('created')) not in (int,float) or type(grant.get('expires')) not in (int,float)
                or not grant['created'] <= self.clock() < grant['expires'] <= grant['created']+TTL
                or type(grant.get('lease_until')) not in (float,int)
                or not self.clock() < grant['lease_until'] <= min(grant['expires'],self.clock()+AUTH_LEASE)
                or (owner is not None and grant.get('owner')!=owner)):
            raise LiveBufferError('LIVE_RESEARCH_REVOKED_OR_EXPIRED')
        return grant

    def renew(self, ticket, owner):
        grant=self.check(ticket,owner)
        grant['lease_until']=min(grant['expires'],self.clock()+AUTH_LEASE)
        self.write(ticket,'.grant.json',grant,replace=True)

    def capture(self, ticket):
        grant=self.check(ticket)
        capture=self.read(ticket,'.capture.json',MAX_CAPTURE)
        if (capture.get('ticket')!=ticket or capture.get('owner')!=grant['owner']
                or digest(canonical(capture))!=grant['capture_sha256']):
            raise LiveBufferError('LIVE_RESEARCH_CAPTURE_CHANGED')
        return capture

    def revoke(self, ticket, owner=None, cancel=None):
        grant=self.read(ticket,'.grant.json',MAX_RESULT)
        if ((owner is not None and owner!=grant['owner']) or
                (cancel is not None and not hmac.compare_digest(digest(cancel.encode()),grant['cancel_sha256']))):
            raise LiveBufferError('LIVE_RESEARCH_NOT_FOUND')
        grant['active']=False
        self.write(ticket,'.grant.json',grant,replace=True)
        self.delete_capture(ticket)
        self.path(ticket,'.result.json').unlink(missing_ok=True)

    def delete_capture(self, ticket):
        self.path(ticket,'.capture.json').unlink(missing_ok=True)

    def result(self, ticket, owner, cancel):
        grant=self.check(ticket,owner)
        if type(cancel) is not str or not hmac.compare_digest(digest(cancel.encode()),grant['cancel_sha256']):
            raise LiveBufferError('LIVE_RESEARCH_NOT_FOUND')
        path=self.path(ticket,'.result.json')
        if not path.exists():return None
        result=self.read(ticket,'.result.json',MAX_RESULT)
        self.check(ticket,owner)
        if result.get('ticket')!=ticket or result.get('owner')!=owner:
            raise LiveBufferError('LIVE_RESEARCH_NOT_FOUND')
        return result

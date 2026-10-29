"""Bounded general-media candidate decoder. Research profile, NOT release approval.

FFmpeg decodes local compressed streams; OpenCV applies the producer RGB/linear
resize operation to selected BGR pixels. Decoder-build differences are measured,
never concealed behind a fabricated TRAIN/export or shared_clock_pts_v1 stamp.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from threading import Event, Lock, Thread

import cv2
import numpy as np
import soundfile as sf

from deployment.runtime import (DeploymentConfig, DeploymentError, ReadinessReport,
    SUPPORTED_MEDIA_PROFILES, validate_upload)
from fusion.encoder_interfaces import Artifact

PROFILE = "aval.ffmpeg-cfr-bgr-cv-linear.v1.NOT_APPROVED"
MAX_WIDTH, MAX_HEIGHT, MAX_FPS = 1920, 1080, 120
MAX_OUTPUT = 512 * 1024
MAX_RAW_BYTES = MAX_WIDTH * MAX_HEIGHT * 3 * 10
WALL_SECONDS, CPU_SECONDS, ADDRESS_BYTES = 20, 15, 1024**3


def require_research_sandbox():
    """Fail closed outside the resource-limited, no-network Linux research container."""
    try:
        status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
        memory = int(Path('/sys/fs/cgroup/memory.max').read_text())
        pids = int(Path('/sys/fs/cgroup/pids.max').read_text())
        quota, period = map(int, Path('/sys/fs/cgroup/cpu.max').read_text().split())
        mounts = [line.split() for line in Path('/proc/mounts').read_text().splitlines()]
        routes = Path('/proc/net/route').read_text().splitlines()[1:]
        ipv6_routes = Path('/proc/net/ipv6_route').read_text().splitlines()
        if (os.getuid() == 0 or int(status['NoNewPrivs']) != 1 or int(status['CapEff'], 16) != 0
                or not 0 < memory <= 4 * 1024**3 or not 0 < pids <= 128
                or not 0 < quota / period <= 2
                or not any(m[1] == '/' and 'ro' in m[3].split(',') for m in mounts)
                or any(r.split() and r.split()[0] != 'lo' for r in routes)
                or any(r.split() and r.split()[-1] != 'lo' for r in ipv6_routes)):
            raise ValueError
        return {'memory_bytes': memory, 'pids': pids, 'cpu_quota': quota / period,
                'uid': os.getuid(), 'network': 'loopback-only', 'read_only_root': True, 'no_new_privileges': True}
    except Exception:
        raise DeploymentError('GENERAL_RESEARCH_SANDBOX_REQUIRED') from None


def bounded(command):
    """No shell, no server secrets in the child, deadline and combined diagnostic cap."""
    try:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env={'PATH':'/usr/bin:/bin', 'LANG':'C', 'LC_ALL':'C'}, close_fds=True)
    except OSError:
        raise DeploymentError('GENERAL_DECODER_UNAVAILABLE') from None
    buffers = [bytearray(), bytearray()]
    lock, overflow = Lock(), Event()
    count = [0]

    def read(i, pipe):
        try:
            while chunk := pipe.read(8192):
                with lock:
                    count[0] += len(chunk)
                    if count[0] > MAX_OUTPUT:
                        overflow.set()
                        process.kill()
                    elif not overflow.is_set():
                        buffers[i].extend(chunk)
        except (OSError, ValueError):
            return

    threads = [Thread(target=read, args=(i, p), daemon=True) for i, p in enumerate((process.stdout, process.stderr))]
    for t in threads:
        t.start()
    try:
        try:
            code = process.wait(timeout=WALL_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
            raise DeploymentError('GENERAL_DECODE_TIMEOUT') from None
        finally:
            for t in threads:
                t.join(timeout=1)
        if overflow.is_set():
            raise DeploymentError('GENERAL_DECODE_OUTPUT_LIMIT')
        if any(t.is_alive() for t in threads):
            raise DeploymentError('GENERAL_DECODE_READER_FAILED')
        if code:
            raise DeploymentError('GENERAL_DECODE_FAILED')
        return bytes(buffers[0])
    finally:
        if process.poll() is None:
            process.kill(); process.wait()
        process.stdout.close(); process.stderr.close()


def fraction(value):
    try:
        result = Fraction(str(value))
        if result.denominator > 10**9:
            raise ValueError
        return result
    except (ValueError, ZeroDivisionError):
        raise DeploymentError('GENERAL_CLOCK_METADATA_INVALID') from None


def clock_plan(metadata, frames):
    """CFR/native-tick checks, not historical shared-clock or numerical acceptance."""
    try:
        streams = metadata['streams']
        if len(streams) != 2 or {s['codec_type'] for s in streams} != {'audio', 'video'}:
            raise DeploymentError('MISSING_OR_AMBIGUOUS_STREAMS')
        video = next(s for s in streams if s['codec_type'] == 'video')
        audio = next(s for s in streams if s['codec_type'] == 'audio')
        if any(fraction(x) != 0 for x in (metadata['format']['start_time'], video['start_time'], audio['start_time'])):
            raise DeploymentError('MODALITY_CLOCK_MISMATCH')
        w, h = video['width'], video['height']
        if (type(w) is not int or type(h) is not int or not 1 <= w <= MAX_WIDTH or not 1 <= h <= MAX_HEIGHT
                or video['pix_fmt'] != 'yuv420p' or video.get('sample_aspect_ratio', '1:1') not in ('1:1', 'N/A')
                or any(s.get('rotation', 0) != 0 for s in video.get('side_data_list', []))
                or str(video.get('tags', {}).get('rotate', '0')) not in ('0','0.0')
                or video.get('color_transfer') in ('smpte2084','arib-std-b67')):
            raise DeploymentError('GENERAL_VIDEO_PROFILE_UNSUPPORTED')
        fps = fraction(video['r_frame_rate'])
        if not 1 <= fps <= MAX_FPS or fraction(video['avg_frame_rate']) != fps:
            raise DeploymentError('GENERAL_VFR_UNSUPPORTED')
        sr = int(audio['sample_rate'])
        if sr < 8000 or sr > 96000 or audio['channels'] not in (1, 2):
            raise DeploymentError('GENERAL_AUDIO_PROFILE_UNSUPPORTED')
        ticks = {s['codec_type']: fraction(s['time_base']) for s in streams}
        if any(t <= 0 or t > Fraction(1, 1000) for t in ticks.values()):
            raise DeploymentError('GENERAL_CLOCK_RESOLUTION_UNSUPPORTED')
        vf = [f for f in frames if f['media_type'] == 'video']
        af = [f for f in frames if f['media_type'] == 'audio']
        if len(vf) < 10 or len(vf) > 1206 or not af or len(af) > 2000 or len(vf)+len(af) != len(frames):
            raise DeploymentError('GENERAL_FRAME_COUNT_INVALID')
        pts = []
        for n, f in enumerate(vf):
            t = int(f['best_effort_timestamp']) * ticks['video']
            if (f['width'] != w or f['height'] != h or f['pix_fmt'] != 'yuv420p'
                    or abs(t - Fraction(n, 1)/fps) > ticks['video']/2):
                raise DeploymentError('GENERAL_VIDEO_CLOCK_MISMATCH')
            pts.append(t)
        if not Fraction(10) <= Fraction(len(vf), 1)/fps <= Fraction('10.05'):
            raise DeploymentError('GENERAL_VIDEO_INCOMPLETE')
        samples = 0
        for f in af:
            t = int(f['best_effort_timestamp']) * ticks['audio']
            n = f['nb_samples']
            if type(n) is not int or not 0 < n <= 8192 or abs(t - Fraction(samples, sr)) > ticks['audio']/2:
                raise DeploymentError('GENERAL_AUDIO_CLOCK_MISMATCH')
            samples += n
        if samples < sr * 10:
            raise DeploymentError('GENERAL_AUDIO_INCOMPLETE')
        if samples > math.ceil(Fraction('10.05')*sr) + af[-1]['nb_samples']:
            raise DeploymentError('GENERAL_AUDIO_CLOCK_MISMATCH')
        indices = [int(sec * fps) for sec in range(10)]
        if len(set(indices)) != 10 or max(indices) >= len(vf):
            raise DeploymentError('GENERAL_VIDEO_INCOMPLETE')
        return {'fps':str(fps), 'width':w, 'height':h, 'video_frames':len(vf), 'native_sample_rate':sr,
            'audio_decoded_samples':samples, 'video_time_base':str(ticks['video']), 'audio_time_base':str(ticks['audio']),
            'frame_indices':indices, 'achieved_frame_pts':[str(pts[i]) for i in indices],
            'achieved_frame_seconds':[float(pts[i]) for i in indices],
            'frame_offset_seconds':[float(pts[i]-sec) for sec,i in enumerate(indices)],
            'clock_policy':'CFR and contiguous audio on zero origin; nearest native tick, no gap repair',
            'historical_shared_clock_approved':False, 'repairs_applied':False}
    except DeploymentError:
        raise
    except (ValueError, KeyError, TypeError, OverflowError, StopIteration):
        raise DeploymentError('GENERAL_CLOCK_METADATA_INVALID') from None


@dataclass(frozen=True)
class DecodedWindow:
    waveform: np.ndarray
    frames: np.ndarray
    source_sha256: str
    timing: dict
    waveform_sha256: str
    frames_sha256: str


class GeneralDecoder:
    def __init__(self, config: DeploymentConfig):
        self.sandbox = require_research_sandbox()
        self.config = config
        self.executables = {}
        for name in ('ffmpeg', 'ffprobe', 'prlimit'):
            path = shutil.which(name)
            if not path:
                raise DeploymentError('GENERAL_DECODER_UNAVAILABLE')
            self.executables[name] = Artifact(path, hashlib.sha256(Path(path).read_bytes()).hexdigest())

    def run(self, name, args):
        for artifact in self.executables.values():
            artifact.verify()
        return bounded([self.executables['prlimit'].path, '--as='+str(ADDRESS_BYTES),
            '--cpu='+str(CPU_SECONDS), '--fsize='+str(64*1024**2), '--nofile=64', '--',
            self.executables[name].path, *args])

    @staticmethod
    def input_flags(suffix):
        # Force the extension's allowed demuxer. MOV external data references
        # and absolute alias paths are disabled, and no network protocol exists.
        flags = ['-protocol_whitelist','file','-format_whitelist','mov,matroska,webm', '-threads','1']
        return flags + (['-f','mov','-enable_drefs','0','-use_absolute_path','0'] if suffix in ('.mp4','.mov')
                        else ['-f','matroska'])

    def decode(self, content: bytes, filename: str) -> DecodedWindow:
        # Existing admission performs path/name/size/codec/duration checks first.
        # Its probe also runs under the isolated, limited candidate process.
        suffix = Path(filename).suffix.lower() if type(filename) is str else ''
        if suffix not in SUPPORTED_MEDIA_PROFILES:
            raise DeploymentError('UNSUPPORTED_EXTENSION')
        flags = self.input_flags(suffix)

        def probe(path):
            from deployment.runtime import MediaInfo, MediaStream
            data = json.loads(self.run('ffprobe', ['-v','error','-max_alloc','67108864', *flags,
                '-show_entries','format=format_name,start_time:format_tags=major_brand,compatible_brands:'
                'stream=index,codec_type,codec_name,duration,start_time,width,height,r_frame_rate,avg_frame_rate,'
                'pix_fmt,sample_aspect_ratio,color_transfer,time_base,sample_rate,channels:stream_tags=rotate:stream_side_data=rotation',
                '-of','json',str(path)]))
            self_metadata.append(data)
            return MediaInfo(tuple(data['format']['format_name'].split(',')),
                tuple((k.lower(), v.lower()) for k,v in data['format'].get('tags',{}).items()),
                tuple(MediaStream(s['codec_type'],s['codec_name'],float(s['duration'])) for s in data['streams']))

        self_metadata = []
        result = validate_upload(content, filename=filename, config=self.config,
            readiness=ReadinessReport('MODEL_NOT_READY', (), False), probe=probe)
        if result.get('error_code') != 'RAW_INFERENCE_GATE_CLOSED':
            raise DeploymentError(result.get('error_code', 'GENERAL_ADMISSION_FAILED'))
        meta = self_metadata[0]
        # Bound geometry/rate before full frame scanning or pixel decoding.
        v = next(s for s in meta['streams'] if s['codec_type']=='video')
        if not 1 <= v['width'] <= MAX_WIDTH or not 1 <= v['height'] <= MAX_HEIGHT or not 1 <= fraction(v['r_frame_rate']) <= MAX_FPS:
            raise DeploymentError('GENERAL_VIDEO_PROFILE_UNSUPPORTED')
        with tempfile.TemporaryDirectory(prefix='aval-general-') as directory:
            root = Path(directory)
            source = root/('source'+suffix); source.write_bytes(content)
            source_hash = hashlib.sha256(content).hexdigest()
            scan = json.loads(self.run('ffprobe', ['-v','error','-max_alloc','67108864',*flags,
                '-show_frames','-show_entries','frame=media_type,best_effort_timestamp,nb_samples,width,height,pix_fmt',
                '-of','json',str(source)]))
            timing = clock_plan(meta, scan['frames'])
            wav, pixels = root/'audio.wav', root/'video.bgr'
            common = ['-nostdin','-v','error','-xerror','-y','-max_alloc','67108864', *flags,
                      '-noautorotate','-i',str(source)]
            self.run('ffmpeg', [*common,'-map','0:a:0','-vn','-ac','1','-ar','16000','-t','10',
                               '-c:a','pcm_s16le','-fs','322048',str(wav)])
            if not wav.is_file() or wav.stat().st_size > 322048:
                raise DeploymentError('GENERAL_AUDIO_INCOMPLETE')
            wave, rate = sf.read(wav, dtype='float32', always_2d=False)
            if rate != 16000 or wave.shape != (160000,) or not np.isfinite(wave).all() or (np.abs(wave)>1).any():
                raise DeploymentError('GENERAL_AUDIO_INCOMPLETE')
            select = '+'.join(f'eq(n\\,{i})' for i in timing['frame_indices'])
            size = timing['width']*timing['height']*3*10
            self.run('ffmpeg', [*common,'-map','0:v:0','-an','-vf','select='+select,'-vsync','0',
                '-frames:v','10','-threads','1','-filter_threads','1','-pix_fmt','bgr24','-f','rawvideo',
                '-fs',str(size+1),str(pixels)])
            if not pixels.is_file() or pixels.stat().st_size != size or size > MAX_RAW_BYTES:
                raise DeploymentError('GENERAL_VIDEO_INCOMPLETE')
            bgr = np.frombuffer(pixels.read_bytes(),np.uint8).reshape(10,timing['height'],timing['width'],3)
            frames = np.stack([cv2.resize(cv2.cvtColor(f,cv2.COLOR_BGR2RGB),(224,224),interpolation=cv2.INTER_LINEAR) for f in bgr])
            if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
                raise DeploymentError('GENERAL_SOURCE_MODIFIED')
        return DecodedWindow(wave, frames, source_hash, timing,
            hashlib.sha256(wave.tobytes()).hexdigest(), hashlib.sha256(frames.tobytes()).hexdigest())

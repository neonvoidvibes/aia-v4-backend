from __future__ import annotations
import subprocess, shlex, os

def to_mono16k_pcm(src_path: str, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cmd = f'ffmpeg -y -loglevel error -i {shlex.quote(src_path)} -ar 16000 -ac 1 -acodec pcm_s16le {shlex.quote(dst_path)}'
    subprocess.check_call(cmd, shell=True)


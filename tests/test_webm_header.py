import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.webm_header import extract_webm_header


def test_extract_header_prefix():
    header = b"\x1a\x45\xdf\xa3\x42\x86\x81\x01"
    cluster = b"\x1fC\xb6u\x10\x20\x30"
    blob = header + cluster + b"\x00\x01"

    extracted = extract_webm_header(blob)

    assert extracted == header


def test_extract_header_returns_none_without_cluster():
    assert extract_webm_header(b"\x1a\x45\xdf\xa3") is None


def test_extract_header_returns_none_for_cluster_at_start():
    cluster_only = b"\x1fC\xb6u\x10\x20\x30"
    assert extract_webm_header(cluster_only) is None

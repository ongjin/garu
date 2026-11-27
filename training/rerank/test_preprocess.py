"""Preprocessing cache behavior against real input files and source changes."""
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import zipfile
import zlib

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_perceptron as trainer


ARRAY_KEYS = {"ids", "vals", "feat_off", "sent_off", "counts", "srcs"}
ZIP_LOCAL_HEADER_SIZE = 30
ZIP_NAME_LENGTH_OFFSET = 26
DEFLATE_RESERVED_BLOCK = 0b110


def write_jsonl(path, record):
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")


@pytest.fixture
def data_dir(tmp_path):
    write_jsonl(tmp_path / "train.jsonl", {
        "text": "고양이", "morphemes": [["고양이", "NNG"]], "src": "SX",
    })
    write_jsonl(tmp_path / "train_top10.jsonl", {
        "text": "고양이", "candidates": [{"score": 1., "tokens": [["고양이", "NNG", 0, 3]]}],
    })
    return tmp_path


def change_tag(path):
    # Keep size and mtime unchanged: cache identity must use the input bytes.
    stat = path.stat()
    path.write_bytes(path.read_bytes().replace(b"NNG", b"NNP"))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def test_dump_change(data_dir):
    before = trainer.preprocess(data_dir, "train")
    assert before["counts"].tolist() == [[1, 0, 0]]
    change_tag(data_dir / "train_top10.jsonl")

    after = trainer.preprocess(data_dir, "train")

    assert after["counts"].tolist() == [[0, 1, 1]]
    assert not np.array_equal(after["ids"], before["ids"])


def test_gold_change(data_dir):
    trainer.preprocess(data_dir, "train")
    change_tag(data_dir / "train.jsonl")

    assert trainer.preprocess(data_dir, "train")["counts"].tolist() == [[0, 1, 1]]


def test_cache_hit(data_dir, monkeypatch):
    before = trainer.preprocess(data_dir, "train")
    cache = data_dir / "cache_train.npz"
    cache_bytes = cache.read_bytes()

    def unexpected(*args, **kwargs):
        pytest.fail("unchanged cache repeated preprocessing or cache writing")

    monkeypatch.setattr(trainer, "norm", unexpected)
    monkeypatch.setattr(trainer, "extract", unexpected)
    monkeypatch.setattr(trainer.np, "savez_compressed", unexpected)

    after = trainer.preprocess(data_dir, "train")

    assert set(after) == ARRAY_KEYS
    assert after["counts"].tolist() == [[1, 0, 0]]
    for key in ARRAY_KEYS:
        np.testing.assert_array_equal(after[key], before[key], strict=True)
    assert cache.read_bytes() == cache_bytes


def test_legacy_cache(data_dir):
    old = trainer.preprocess(data_dir, "train")
    old["counts"][:] = 99
    np.savez_compressed(data_dir / "cache_train.npz", **old)

    rebuilt = trainer.preprocess(data_dir, "train")

    assert set(rebuilt) == ARRAY_KEYS
    assert rebuilt["counts"].tolist() == [[1, 0, 0]]


def test_cache_closed(data_dir, monkeypatch):
    trainer.preprocess(data_dir, "train")
    load = np.load
    handles = []

    def tracked_load(*args, **kwargs):
        archive = load(*args, **kwargs)
        handles.append(archive)
        return archive

    monkeypatch.setattr(trainer.np, "load", tracked_load)
    trainer.preprocess(data_dir, "train")

    assert handles
    assert all(archive.zip is None for archive in handles)


def fail_write(file, **arrays):
    if hasattr(file, "write"):
        file.write(b"partial archive")
    else:
        Path(file).write_bytes(b"partial archive")
    raise OSError("simulated disk write failure")


def test_failed_cache_write(data_dir, monkeypatch):
    trainer.preprocess(data_dir, "train")
    cache = data_dir / "cache_train.npz"
    old_bytes = cache.read_bytes()
    change_tag(data_dir / "train_top10.jsonl")
    files_before = set(data_dir.iterdir())

    with monkeypatch.context() as patch:
        patch.setattr(trainer.np, "savez_compressed", fail_write)
        with pytest.raises(OSError, match="simulated disk write failure"):
            trainer.preprocess(data_dir, "train")

    assert cache.read_bytes() == old_bytes
    assert set(data_dir.iterdir()) == files_before
    assert trainer.preprocess(data_dir, "train")["counts"].tolist() == [[0, 1, 1]]


def test_failed_first_write(data_dir, monkeypatch):
    files_before = set(data_dir.iterdir())
    monkeypatch.setattr(trainer.np, "savez_compressed", fail_write)

    with pytest.raises(OSError, match="simulated disk write failure"):
        trainer.preprocess(data_dir, "train")

    assert set(data_dir.iterdir()) == files_before


def test_corrupt_cache(data_dir):
    (data_dir / "cache_train.npz").write_bytes(b"PK\x03\x04truncated archive")

    assert trainer.preprocess(data_dir, "train")["counts"].tolist() == [[1, 0, 0]]


def test_corrupt_member(data_dir):
    before = trainer.preprocess(data_dir, "train")
    cache = data_dir / "cache_train.npz"
    with zipfile.ZipFile(cache) as archive:
        member = archive.getinfo("ids.npy")
        assert member.compress_type == zipfile.ZIP_DEFLATED
    payload = bytearray(cache.read_bytes())
    name_len, extra_len = struct.unpack_from(
        "<HH", payload, member.header_offset + ZIP_NAME_LENGTH_OFFSET,
    )
    data_start = member.header_offset + ZIP_LOCAL_HEADER_SIZE + name_len + extra_len
    payload[data_start] |= DEFLATE_RESERVED_BLOCK
    cache.write_bytes(payload)
    with np.load(cache) as archive:
        with pytest.raises(zlib.error, match="invalid block type"):
            archive["ids"]

    rebuilt = trainer.preprocess(data_dir, "train")

    assert rebuilt["counts"].tolist() == [[1, 0, 0]]
    with np.load(cache) as archive:
        for key in ARRAY_KEYS:
            np.testing.assert_array_equal(rebuilt[key], before[key], strict=True)
            np.testing.assert_array_equal(archive[key], before[key], strict=True)


def run_copy(source_dir, data_dir):
    script = (
        "import json, sys; from pathlib import Path; "
        "from train_perceptron import preprocess; "
        "data = preprocess(Path(sys.argv[1]), 'train'); "
        "print(json.dumps({key: value.tolist() for key, value in data.items()}))"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(data_dir)],
        cwd=source_dir, capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout.splitlines()[-1])


@pytest.mark.parametrize(("filename", "change", "key", "expected"), [
    ("features.py", "\ndef hash_feats(feats):\n    return [7], [3.]\n", "ids", [7]),
    ("../gold_testset/expand/ep_norm.py",
     "\ndef normalize_ep_morphemes(tokens):\n    return []\n", "counts", [[0, 0, 0]]),
    ("train_perceptron.py",
     "\ndef prf_counts(pred, gold_counter):\n    return 0, 7, 8\n", "counts", [[0, 7, 8]]),
])
def test_source_change(data_dir, filename, change, key, expected):
    source_dir = data_dir / "source" / "rerank"
    norm_dir = source_dir.parent / "gold_testset" / "expand"
    source_dir.mkdir(parents=True)
    norm_dir.mkdir(parents=True)
    for name in ("train_perceptron.py", "features.py"):
        shutil.copyfile(trainer.BASE / name, source_dir / name)
    shutil.copyfile(trainer.BASE.parent / "gold_testset/expand/ep_norm.py",
                    norm_dir / "ep_norm.py")
    before = run_copy(source_dir, data_dir)
    assert before[key] != expected
    source = source_dir / filename
    source.write_text(source.read_text(encoding="utf-8") + change, encoding="utf-8")

    assert run_copy(source_dir, data_dir)[key] == expected


def test_promise_target_uses_split_output(tmp_path):
    # 후보는 붙은 `ㄹ게요/EF`로 채점되지만 공개 출력과 NIKL 정답은 `ㄹ게/EF + 요/JX`다.
    write_jsonl(tmp_path / "train.jsonl", {
        "text": "할게요", "morphemes": [["하", "VV"], ["ㄹ게", "EF"], ["요", "JX"]], "src": "SX",
    })
    write_jsonl(tmp_path / "train_top10.jsonl", {
        "text": "할게요", "candidates": [{"score": 1., "tokens": [["하", "VV", 0, 3], ["ㄹ게요", "EF", 0, 3]]}],
    })
    assert trainer.preprocess(tmp_path, "train")["counts"].tolist() == [[3, 0, 0]]

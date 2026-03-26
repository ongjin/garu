"""Build a dictionary from Korean Wikipedia titles and merge into GMDL model.

Downloads kowiki title dump, filters for English/mixed terms,
builds FST via Rust tool, and appends as GMDL section 6.

Usage:
    python training/build_wiki_dict.py [--model models/base_v2.gmdl] [--output models/base.gmdl]
"""

import gzip
import os
import re
import struct
import subprocess
import tempfile
import urllib.request
from pathlib import Path

WIKI_TITLES_URL = "https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-all-titles-in-ns0.gz"

EXCLUDE_PREFIXES = [
    "위키백과:", "틀:", "카테고리:", "파일:", "모듈:",
    "도움말:", "사용자:", "미디어위키:", "Wikipedia:",
    "Template:", "Category:", "File:", "Help:", "User:",
]

POS_NNP = 1

HAS_ALPHA = re.compile(r"[A-Za-z]")
PARENS = re.compile(r"\s*\(.*\)\s*$")
NOISE_PATTERNS = re.compile(r"^\d{4}년|^\d+번|^\d+호|^\d+편|^\d+행$|결의 제|의원 선거|지진$")
MAX_TITLE_LEN = 20
MAX_WORDS = 2
MIN_ALPHA = 2
MIN_TITLE_LEN = 3


def download_titles(cache_path: Path) -> list[str]:
    """Download and cache wiki titles."""
    if not cache_path.exists():
        print(f"Downloading {WIKI_TITLES_URL} ...")
        urllib.request.urlretrieve(WIKI_TITLES_URL, cache_path)
        print(f"Saved to {cache_path}")
    else:
        print(f"Using cached {cache_path}")

    titles = []
    with gzip.open(cache_path, "rt", encoding="utf-8") as f:
        for line in f:
            title = line.strip().replace("_", " ")
            if title:
                titles.append(title)
    print(f"Total titles: {len(titles)}")
    return titles


def filter_titles(titles: list[str]) -> list[str]:
    """Filter for English/mixed terms."""
    seen = set()
    result = []

    for title in titles:
        if any(title.startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        if not HAS_ALPHA.search(title):
            continue

        clean = PARENS.sub("", title).strip()
        if not clean or len(clean) < MIN_TITLE_LEN:
            continue
        if len(clean) > MAX_TITLE_LEN or len(clean.split()) > MAX_WORDS:
            continue
        if sum(1 for c in clean if c.isascii() and c.isalpha()) < MIN_ALPHA:
            continue
        if NOISE_PATTERNS.search(clean):
            continue

        if clean in seen:
            continue
        seen.add(clean)
        result.append(clean)

    print(f"Filtered titles: {len(result)}")
    return result


def build_fst_dict(titles: list[str], root: Path) -> bytes:
    """Build FST dict binary via Rust build-dict tool."""
    # Write sorted word list
    sorted_titles = sorted(titles, key=lambda t: t.encode("utf-8"))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for title in sorted_titles:
            f.write(f"{title}\t{POS_NNP}\n")
        wordlist_path = f.name

    dict_path = tempfile.mktemp(suffix=".bin")

    try:
        # Run Rust build-dict tool
        tool_path = root / "target" / "release" / "build-dict"
        if not tool_path.exists():
            print("Building build-dict tool...")
            subprocess.run(
                ["cargo", "build", "-p", "garu-tools", "--release"],
                cwd=root, check=True,
            )

        result = subprocess.run(
            [str(tool_path), wordlist_path, dict_path],
            capture_output=True, text=True,
        )
        print(result.stderr, end="")
        if result.returncode != 0:
            raise RuntimeError(f"build-dict failed: {result.stderr}")

        dict_bytes = Path(dict_path).read_bytes()
        print(f"FST dict size: {len(dict_bytes) / 1024:.1f} KB")
        return dict_bytes
    finally:
        os.unlink(wordlist_path)
        if os.path.exists(dict_path):
            os.unlink(dict_path)


def merge_dict_into_gmdl(model_path: Path, dict_bytes: bytes, output_path: Path):
    """Append dict as section 6 to GMDL model, removing existing section 6 if present."""
    data = model_path.read_bytes()

    assert data[:4] == b"GMDL", "Not a GMDL file"
    version = struct.unpack_from("<I", data, 4)[0]
    print(f"GMDL version: {version}")

    pos = 8
    sections = []
    while pos < len(data):
        stype = data[pos]
        slen = struct.unpack_from("<I", data, pos + 1)[0]
        section_data = data[pos + 5 : pos + 5 + slen]
        if stype != 6:
            sections.append((stype, section_data))
        else:
            print(f"Removing existing section 6 ({slen} bytes)")
        pos += 5 + slen

    out = bytearray()
    out.extend(b"GMDL")
    out.extend(struct.pack("<I", version))

    for stype, sdata in sections:
        out.extend(struct.pack("B", stype))
        out.extend(struct.pack("<I", len(sdata)))
        out.extend(sdata)

    out.extend(struct.pack("B", 6))
    out.extend(struct.pack("<I", len(dict_bytes)))
    out.extend(dict_bytes)

    output_path.write_bytes(out)
    print(f"Output: {output_path} ({len(out) / 1024 / 1024:.2f} MB)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build wiki dict and merge into GMDL")
    parser.add_argument("--model", default="models/base_v2.gmdl", help="Input GMDL model")
    parser.add_argument("--output", default="models/base.gmdl", help="Output GMDL model")
    parser.add_argument("--cache", default="training/kowiki-titles.gz", help="Cached titles file")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    model_path = root / args.model
    output_path = root / args.output
    cache_path = root / args.cache

    titles = download_titles(cache_path)
    filtered = filter_titles(titles)
    dict_bytes = build_fst_dict(filtered, root)
    merge_dict_into_gmdl(model_path, dict_bytes, output_path)


if __name__ == "__main__":
    main()

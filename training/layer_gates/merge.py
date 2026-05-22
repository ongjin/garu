"""filter 통과 후보를 content_dict.txt에 머지.

format: TSV — `surface\tpos\tfreq`.
같은 (surface, pos) 중복 시 skip.
"""


def parse_dict_line(line: str) -> tuple[str, str, int]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 3:
        raise ValueError(f"invalid line: {line!r}")
    return parts[0], parts[1], int(parts[2])


def append_entries_to_dict(candidates: list[dict], dict_path: str) -> int:
    existing: set[tuple[str, str]] = set()
    with open(dict_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                s, p, _ = parse_dict_line(line)
                existing.add((s, p))
            except ValueError:
                continue

    added = 0
    with open(dict_path, "a") as f:
        for c in candidates:
            key = (c["surface"], c["normalized_pos"])
            if key in existing:
                continue
            f.write(f"{c['surface']}\t{c['normalized_pos']}\t{c['frequency']}\n")
            existing.add(key)
            added += 1
    return added

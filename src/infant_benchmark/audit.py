"""Inspect a local DBL-style audio tree without altering it or inferring identity."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import wave


LABELS = ("eairh", "eh", "heh", "neh", "owh")
FOLDERS = {
    "belly_pain": "eairh",
    "burping": "eh",
    "discomfort": "heh",
    "hungry": "neh",
    "tired": "owh",
    **{label: label for label in LABELS},
}
SUFFIX = re.compile(r"_(eairh|eh|heh|neh|owh)\.wav$", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_dataset(root: Path) -> dict:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")

    folder_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter({label: 0 for label in LABELS})
    format_counts: Counter[str] = Counter()
    files_by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    mismatches: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    metadata: list[str] = []
    other_wav_count = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if path.suffix.lower() == ".csv" and (
            "meta" in path.name.lower() or "manifest" in path.name.lower()
        ):
            metadata.append(relative.as_posix())
        if path.suffix.lower() != ".wav":
            continue
        top_folder = relative.parts[0].lower() if len(relative.parts) > 1 else ""
        expected_label = FOLDERS.get(top_folder)
        if expected_label is None:
            other_wav_count += 1
            continue
        folder_counts[top_folder] += 1
        label_counts[expected_label] += 1
        match = SUFFIX.search(path.name)
        if match and match.group(1).lower() != expected_label:
            mismatches.append(
                {
                    "path": relative.as_posix(),
                    "folder_dbl_label": expected_label,
                    "filename_suffix": match.group(1).lower(),
                }
            )
        try:
            digest = _sha256(path)
            files_by_hash[digest].append(
                {"path": relative.as_posix(), "folder_dbl_label": expected_label}
            )
        except OSError as exc:
            errors.append({"path": relative.as_posix(), "operation": "sha256", "error": str(exc)})
            continue
        try:
            with wave.open(str(path), "rb") as audio:
                key = f"{audio.getframerate()}Hz/{audio.getnchannels()}ch/{audio.getsampwidth() * 8}bit"
                format_counts[key] += 1
        except (OSError, EOFError, wave.Error) as exc:
            errors.append({"path": relative.as_posix(), "operation": "wave_header", "error": str(exc)})

    duplicates = {digest: files for digest, files in files_by_hash.items() if len(files) > 1}
    cross = [
        {"sha256": digest, "files": files}
        for digest, files in sorted(duplicates.items())
        if len({item["folder_dbl_label"] for item in files}) > 1
    ]
    return {
        "dataset_root": str(root),
        "read_only": True,
        "classified_wav_count": sum(label_counts.values()),
        "other_wav_count": other_wav_count,
        "counts_by_folder": dict(sorted(folder_counts.items())),
        "counts_by_dbl_label": {label: label_counts[label] for label in LABELS},
        "missing_dbl_labels": [label for label in LABELS if not label_counts[label]],
        "audio_formats": dict(sorted(format_counts.items())),
        "metadata_files": metadata,
        "identity_status": "not_verified",
        "tier_status": "not_verified",
        "filename_suffix_disagreement_count": len(mismatches),
        "filename_suffix_disagreements": mismatches,
        "exact_duplicate_hash_group_count": len(duplicates),
        "cross_label_duplicate_group_count": len(cross),
        "cross_label_duplicate_groups": cross,
        "read_errors": errors,
        "interpretation": (
            "Folder labels are provisional mappings for inventory, not verified labels. "
            "Filename suffix discrepancies are flags, not corrections. SHA-256 compares "
            "entire WAV files byte-for-byte and cannot rule out acoustically similar files. "
            "No infant identities, dataset tiers, or splits are inferred."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Read-only WAV dataset directory")
    args = parser.parse_args(argv)
    try:
        report = inspect_dataset(args.root)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 3 if report["read_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

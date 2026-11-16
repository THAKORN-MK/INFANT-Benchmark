"""Read-only audit of a released two-tier metadata.csv, not a train command."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

from infant_benchmark.audit import LABELS

TIER_ALIAS = {'1214 COLLCTED DATA': '1214 Collected Data'}


def resolve_metadata_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    if '\\' in relative:
        raise ValueError('Metadata paths must use forward slashes')
    path = PurePosixPath(relative)
    parts = path.parts
    if not parts or path.is_absolute() or any(p in ('..', '.') for p in parts) or ':' in parts[0]:
        raise ValueError(f'Unsafe relative path: {relative}')
    original = (root / Path(*parts)).resolve()
    if not original.is_relative_to(root):
        raise ValueError(f'Path escapes dataset root: {relative}')
    if original.exists() or parts[0] not in TIER_ALIAS:
        return original
    mapped = (root / TIER_ALIAS[parts[0]] / Path(*parts[1:])).resolve()
    if not mapped.is_relative_to(root):
        raise ValueError(f'Mapped path escapes dataset root: {relative}')
    return mapped


def inspect_manifest(root: Path) -> dict:
    root = root.resolve()
    with (root / 'metadata.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'relative_path', 'tier_name', 'class', 'clip_id'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('metadata.csv missing required columns')
        rows = list(reader)
    hashes = defaultdict(list)
    missing, errors, mismatches, resolved_paths = [], [], [], set()
    counts = Counter()
    for row in rows:
        rel = row['relative_path']
        label = row['class']
        counts[row['tier_name']] += 1
        try:
            file = resolve_metadata_path(root, rel)
            if file in resolved_paths:
                errors.append({'path': rel, 'reason': 'duplicate metadata path'})
            resolved_paths.add(file)
            if label not in LABELS or file.parent.name.lower() != label:
                mismatches.append({'path': rel, 'label': label, 'parent': file.parent.name})
            if not file.is_file():
                missing.append(rel)
                continue
            digest = hashlib.sha256()
            with file.open('rb') as audio:
                for block in iter(lambda: audio.read(1024 * 1024), b''):
                    digest.update(block)
            hashes[digest.hexdigest()].append({'path': rel, 'label': label, 'tier': row['tier_name']})
        except (OSError, ValueError) as exc:
            errors.append({'path': rel, 'reason': str(exc)})
    duplicates = [v for v in hashes.values() if len(v) > 1]
    cross_label = [v for v in duplicates if len({r['label'] for r in v}) > 1]
    cross_tier = [v for v in duplicates if len({r['tier'] for r in v}) > 1]
    return {
        'dataset_root': str(root), 'metadata_rows': len(rows), 'tier_counts': dict(counts),
        'tier_alias': TIER_ALIAS, 'missing_count': len(missing), 'missing_paths': missing,
        'label_folder_mismatches': mismatches, 'errors': errors,
        'duplicate_hash_groups': len(duplicates), 'cross_label_hash_groups': len(cross_label),
        'cross_tier_hash_groups': len(cross_tier), 'cross_label_details': cross_label,
        'basic_integrity_pass': bool(rows) and not (missing or errors or mismatches or cross_label),
        'ready_for_training': False,
        'identity_status': 'not_verified',
        'warning': 'Read-only byte-level audit; successful checks do not prove identity, independence, or protocol validity.'
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    try:
        result = inspect_manifest(args.root)
    except (ValueError, OSError) as exc:
        p.exit(2, f'error: {exc}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['basic_integrity_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

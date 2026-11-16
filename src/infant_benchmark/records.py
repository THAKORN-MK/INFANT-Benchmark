"""Read one corpus tier; identity grouping is intentionally not inferred."""
import csv
import hashlib
from pathlib import Path
from infant_benchmark.audit import LABELS, SUFFIX
from infant_benchmark.manifest import resolve_metadata_path


def load_records(root: Path, tier: str) -> list[dict]:
    if tier not in ('Dunstan-Core', 'InfantCry-1214'):
        raise ValueError('Select exactly one known corpus tier')
    root = root.resolve()
    seen, records = set(), []
    with (root / 'metadata.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'relative_path', 'tier_name', 'class', 'clip_id'}.issubset(reader.fieldnames or []):
            raise ValueError('Missing metadata columns')
        for row in reader:
            if row['tier_name'] != tier:
                continue
            path = resolve_metadata_path(root, row['relative_path'])
            label = row['class']
            if label not in LABELS or path.parent.name.lower() != label or not path.is_file() or path in seen:
                raise ValueError(f'Invalid, duplicate, absent or mismatched record: {row["relative_path"]}')
            seen.add(path)
            digest = hashlib.sha256()
            with path.open('rb') as audio:
                for block in iter(lambda: audio.read(1024 * 1024), b''):
                    digest.update(block)
            records.append({'path': path, 'relative_path': row['relative_path'], 'label': label,
                            'target': LABELS.index(label), 'sha256': digest.hexdigest(), 'clip_id': row['clip_id']})
    if not records:
        raise ValueError(f'No records for tier: {tier}')
    return records


def suffix_disagreements(records):
    issues = []
    for row in records:
        match = SUFFIX.search(row['path'].name)
        if match and match.group(1).lower() != row['label']:
            issues.append({'path': row['relative_path'], 'folder_label': row['label'],
                           'filename_suffix': match.group(1).lower()})
    return issues

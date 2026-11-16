import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))


class ManifestTests(unittest.TestCase):
    def test_resolves_explicit_tier_alias_and_rejects_path_escape(self):
        from infant_benchmark.manifest import resolve_metadata_path
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / '1214 Collected Data' / 'eh' / 'a.wav'
            target.parent.mkdir(parents=True)
            target.write_bytes(b'example')
            self.assertEqual(resolve_metadata_path(root, '1214 COLLCTED DATA/eh/a.wav'), target.resolve())
            with self.assertRaises(ValueError):
                resolve_metadata_path(root, '../escape.wav')

    def test_audit_rejects_cross_class_duplicate_without_modifying_inputs(self):
        from infant_benchmark.manifest import inspect_manifest
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('eh', 'heh'):
                p = root / '1214 Collected Data' / name / 'a.wav'
                p.parent.mkdir(parents=True)
                p.write_bytes(b'exactly-the-same')
            metadata = root / 'metadata.csv'
            with metadata.open('w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=['relative_path', 'tier_name', 'class', 'clip_id'])
                w.writeheader()
                for name in ('eh', 'heh'):
                    w.writerow({'relative_path': f'1214 COLLCTED DATA/{name}/a.wav', 'tier_name':'InfantCry-1214','class':name,'clip_id':'1'})
            result = inspect_manifest(root)
            self.assertEqual(result['cross_label_hash_groups'], 1)
            self.assertEqual(result['missing_count'], 0)
            self.assertFalse(result['ready_for_training'])
            self.assertEqual((root / '1214 Collected Data' / 'eh' / 'a.wav').read_bytes(), b'exactly-the-same')

    def test_missing_metadata_file_blocks_readiness(self):
        from infant_benchmark.manifest import inspect_manifest
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'metadata.csv').write_text('relative_path,tier_name,class,clip_id\nmissing/eh/a.wav,InfantCry-1214,eh,1\n', encoding='utf-8')
            result = inspect_manifest(root)
            self.assertEqual(result['missing_count'], 1)
            self.assertFalse(result['ready_for_training'])

    def test_clean_manifest_alone_cannot_claim_training_readiness(self):
        from infant_benchmark.manifest import inspect_manifest
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'tier' / 'eh' / 'a.wav'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'input')
            (root / 'metadata.csv').write_text('relative_path,tier_name,class,clip_id\ntier/eh/a.wav,tier,eh,1\n', encoding='utf-8')
            result = inspect_manifest(root)
            self.assertTrue(result['basic_integrity_pass'])
            self.assertFalse(result['ready_for_training'])


if __name__ == '__main__':
    unittest.main()

import csv
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


class RecordTests(unittest.TestCase):
    def test_same_audio_gets_same_group_within_selected_tier(self):
        from infant_benchmark.records import load_records
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / '1214 Collected Data' / 'eh'
            folder.mkdir(parents=True)
            (folder / 'a.wav').write_bytes(b'example')
            (folder / 'b.wav').write_bytes(b'example')
            with (root / 'metadata.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=['relative_path', 'tier_name', 'class', 'clip_id'])
                writer.writeheader()
                for f in ('a.wav', 'b.wav'):
                    writer.writerow({'relative_path': f'1214 COLLCTED DATA/eh/{f}', 'tier_name':'InfantCry-1214', 'class':'eh', 'clip_id':f})
            result = load_records(root, 'InfantCry-1214')
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]['sha256'], result[1]['sha256'])

    def test_missing_file_or_wrong_label_is_rejected(self):
        from infant_benchmark.records import load_records
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'metadata.csv').write_text('relative_path,tier_name,class,clip_id\nx/eh/no.wav,InfantCry-1214,heh,1\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_records(root, 'InfantCry-1214')

    def test_unknown_tier_is_rejected(self):
        from infant_benchmark.records import load_records
        with self.assertRaises(ValueError):
            load_records(Path('.'), 'combined')

    def test_filename_suffix_disagreement_is_reported_not_relabelled(self):
        from infant_benchmark.records import suffix_disagreements
        records = [{'path': Path('/tier/eh/000543_M_heh.wav'), 'label': 'eh', 'relative_path': 'eh/000543_M_heh.wav'}]
        issues = suffix_disagreements(records)
        self.assertEqual(len(issues), 1)
        self.assertEqual(records[0]['label'], 'eh')
        self.assertEqual(issues[0]['filename_suffix'], 'heh')


if __name__ == '__main__':
    unittest.main()

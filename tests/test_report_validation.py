import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

LABELS = ('eairh', 'eh', 'heh', 'neh', 'owh')


def fixture():
    records = [{'label': label, 'sha256': f'{side}-{label}'} for side in ('train', 'test') for label in LABELS]
    split = {'train': list(range(5)), 'heldout': list(range(5, 10))}
    rows = []
    for i, label in enumerate(LABELS):
        row = {'record_index': str(i + 5), 'true_label': label, 'predicted_label': label}
        row.update({f'prob_{name}': '1.0' if name == label else '0.0' for name in LABELS})
        rows.append(row)
    return records, split, rows


class ReportValidationTests(unittest.TestCase):
    def test_perfect_predictions_are_recomputed(self):
        from infant_benchmark.reporting import validate_fold
        records, split, rows = fixture()
        result = validate_fold(records, split, rows)
        self.assertEqual(result['metrics']['accuracy'], 1.0)
        self.assertEqual(result['metrics']['macro_f1'], 1.0)
        self.assertEqual(result['metrics']['macro_auc_ovr'], 1.0)

    def test_group_overlap_is_rejected(self):
        from infant_benchmark.reporting import validate_fold
        records, split, rows = fixture()
        records[5]['sha256'] = records[0]['sha256']
        with self.assertRaisesRegex(ValueError, 'hash'):
            validate_fold(records, split, rows)

    def test_wrong_or_duplicate_prediction_id_is_rejected(self):
        from infant_benchmark.reporting import validate_fold
        records, split, rows = fixture()
        rows[1]['record_index'] = rows[0]['record_index']
        with self.assertRaises(ValueError):
            validate_fold(records, split, rows)

    def test_prediction_label_must_match_probabilities(self):
        from infant_benchmark.reporting import validate_fold
        records, split, rows = fixture()
        rows[0]['predicted_label'] = 'eh'
        with self.assertRaises(ValueError):
            validate_fold(records, split, rows)

    def test_report_generator_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'report').mkdir()
            marker = root / 'report' / 'keep.txt'
            marker.write_text('keep', encoding='utf-8')
            script = Path(__file__).resolve().parents[1] / 'scripts' / 'build_run_report.py'
            result = subprocess.run([sys.executable, str(script), '--run', str(root / 'run'),
                                     '--dataset-root', str(root / 'data'), '--source-root', str(root / 'src'),
                                     '--output', str(root / 'report')], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Refusing to overwrite', result.stderr)
            self.assertEqual(marker.read_text(encoding='utf-8'), 'keep')


if __name__ == '__main__':
    unittest.main()

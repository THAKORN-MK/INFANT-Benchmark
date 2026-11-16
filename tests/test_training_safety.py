import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


class TrainingSafetyTests(unittest.TestCase):
    def test_output_inside_dataset_is_rejected(self):
        from infant_benchmark.training_safety import check_output_location
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'dataset'
            root.mkdir()
            with self.assertRaises(ValueError):
                check_output_location(root, root / 'runs')

    def test_output_outside_dataset_is_allowed(self):
        from infant_benchmark.training_safety import check_output_location
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'dataset'
            root.mkdir()
            check_output_location(root, Path(tmp) / 'runs')


if __name__ == '__main__':
    unittest.main()

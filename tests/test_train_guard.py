import os
from pathlib import Path
import subprocess
import sys
import unittest

PROJECT = Path(__file__).resolve().parents[1]


class TrainGuardTests(unittest.TestCase):
    def test_training_command_defaults_to_dry_run(self):
        env = os.environ.copy()
        env['PYTHONPATH'] = str(PROJECT / 'src')
        process = subprocess.run([sys.executable, '-m', 'infant_benchmark.train', '--dataset-root', str(PROJECT / 'data')],
                                 capture_output=True, text=True, env=env)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn('dry run', process.stdout.lower())
        self.assertNotIn('epoch 1', process.stdout.lower())

    def test_execute_without_approximation_acknowledgement_is_rejected(self):
        env = os.environ.copy()
        env['PYTHONPATH'] = str(PROJECT / 'src')
        process = subprocess.run([sys.executable, '-m', 'infant_benchmark.train', '--dataset-root', str(PROJECT / 'data'), '--execute'],
                                 capture_output=True, text=True, env=env)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn('acknowledge', process.stderr.lower())

    def test_invalid_dataset_execution_fails_without_creating_run(self):
        import tempfile
        env = os.environ.copy()
        env['PYTHONPATH'] = str(PROJECT / 'src')
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / 'runs'
            result = subprocess.run([sys.executable, '-m', 'infant_benchmark.train',
                                     '--dataset-root', str(Path(tmp) / 'missing'),
                                     '--output-root', str(run_root), '--execute',
                                     '--acknowledge-approximation', '--acknowledge-data-issues'],
                                    capture_output=True, text=True, env=env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('dataset directory', result.stderr.lower())
            self.assertFalse(run_root.exists())

    def test_execute_requires_data_issues_acknowledgement(self):
        env = os.environ.copy()
        env['PYTHONPATH'] = str(PROJECT / 'src')
        result = subprocess.run([sys.executable, '-m', 'infant_benchmark.train', '--dataset-root', str(PROJECT / 'data'),
                                 '--execute', '--acknowledge-approximation'], capture_output=True, text=True, env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('data-issues', result.stderr.lower())


if __name__ == '__main__':
    unittest.main()

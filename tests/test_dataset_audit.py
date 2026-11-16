import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import wave


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"


def write_wave(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(struct.pack("<" + "h" * len(values), *values))


def invoke(root):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    return subprocess.run(
        [sys.executable, "-m", "infant_benchmark.audit", "--root", str(root)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        check=False,
    )


class DatasetAuditTests(unittest.TestCase):
    def test_audit_reports_exact_cross_label_duplicate_and_filename_disagreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_wave(root / "burping" / "001_M_heh.wav", [1, -1, 2, -2])
            write_wave(root / "discomfort" / "001_M_heh.wav", [1, -1, 2, -2])
            write_wave(root / "not_baby" / "ambient.wav", [0, 1, 2, 3])
            (root / "metadata.csv").write_text("id\n001\n", encoding="utf-8")
            before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}

            result = invoke(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["classified_wav_count"], 2)
            self.assertEqual(report["other_wav_count"], 1)
            self.assertEqual(report["counts_by_dbl_label"]["eh"], 1)
            self.assertEqual(report["counts_by_dbl_label"]["heh"], 1)
            self.assertEqual(report["cross_label_duplicate_group_count"], 1)
            self.assertEqual(report["filename_suffix_disagreement_count"], 1)
            self.assertEqual(report["identity_status"], "not_verified")
            self.assertIn("metadata.csv", report["metadata_files"])
            self.assertEqual(report["read_errors"], [])
            after = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before, after)

    def test_no_files_is_reported_without_claiming_a_verified_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = invoke(tmp)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["classified_wav_count"], 0)
            self.assertEqual(report["identity_status"], "not_verified")
            self.assertEqual(report["missing_dbl_labels"], ["eairh", "eh", "heh", "neh", "owh"])

    def test_missing_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = invoke(Path(tmp) / "does-not-exist")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("does not exist", result.stderr)

    def test_invalid_wav_header_is_reported_and_sets_failure_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hungry" / "broken.WAV"
            path.parent.mkdir()
            path.write_bytes(b"not a valid wave file")
            result = invoke(tmp)
            self.assertEqual(result.returncode, 3)
            report = json.loads(result.stdout)
            self.assertEqual(report["classified_wav_count"], 1)
            self.assertEqual(report["read_errors"][0]["operation"], "wave_header")

    def test_same_label_duplicate_does_not_become_cross_label_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_wave(root / "hungry" / "a_neh.wav", [12, -12])
            write_wave(root / "hungry" / "b_neh.wav", [12, -12])
            result = invoke(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["exact_duplicate_hash_group_count"], 1)
            self.assertEqual(report["cross_label_duplicate_group_count"], 0)


if __name__ == "__main__":
    unittest.main()

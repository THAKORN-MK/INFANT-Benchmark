import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


class SplitTests(unittest.TestCase):
    def test_hash_duplicate_never_crosses_a_fold(self):
        from infant_benchmark.splits import plan_folds
        labels = [0, 0] + [i % 2 for i in range(20)]
        groups = ['same', 'same'] + [f'g{i}' for i in range(20)]
        folds = plan_folds(labels, groups, seed=42)
        self.assertEqual(len(folds), 5)
        seen = []
        for train, held_out in folds:
            self.assertFalse(set(train) & set(held_out))
            self.assertFalse({groups[i] for i in train} & {groups[i] for i in held_out})
            seen.extend(held_out)
        self.assertEqual(sorted(seen), list(range(len(labels))))

    def test_rejects_conflicting_labels_for_same_hash(self):
        from infant_benchmark.splits import plan_folds
        with self.assertRaises(ValueError):
            plan_folds([0, 1], ['same', 'same'], seed=42)

    def test_rejects_not_enough_groups_per_class(self):
        from infant_benchmark.splits import plan_folds
        with self.assertRaises(ValueError):
            plan_folds([0, 0, 1, 1], ['a', 'b', 'c', 'd'], seed=42)


if __name__ == '__main__':
    unittest.main()

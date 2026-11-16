"""Deterministic stratified folds grouped by exact byte hash, not infant identity."""
from collections import defaultdict
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def plan_folds(labels, groups, *, seed=42):
    y, g = np.asarray(labels), np.asarray(groups)
    if y.ndim != 1 or g.ndim != 1 or len(y) != len(g) or not len(y):
        raise ValueError('labels and groups must be nonempty, aligned 1-D arrays')
    by_group = defaultdict(set)
    by_class = defaultdict(set)
    for label, group in zip(y, g):
        by_group[group].add(label)
        by_class[label].add(group)
    if any(len(values) != 1 for values in by_group.values()):
        raise ValueError('An exact-duplicate group has conflicting class labels')
    if len(by_class) < 2 or any(len(group_set) < 5 for group_set in by_class.values()):
        raise ValueError('Each of at least two classes needs five distinct groups')
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = []
    for train, test in cv.split(np.zeros(len(y)), y, g):
        if set(y[train]) != set(y) or set(y[test]) != set(y):
            raise ValueError('A fold is missing a class; choose a different protocol or seed')
        folds.append((train.tolist(), test.tolist()))
    return folds

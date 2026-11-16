"""Fold-local scaling and MI selection; fit only on the training subset."""
from functools import partial
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_feature_transform(train: np.ndarray, labels: np.ndarray, *, k: int = 100, seed: int = 42) -> Pipeline:
    data = np.asarray(train)
    target = np.asarray(labels)
    if data.ndim != 2 or target.ndim != 1 or len(data) != len(target) or not np.isfinite(data).all():
        raise ValueError('Training features and labels must be finite, aligned arrays')
    if len(data) < 4 or not 1 <= k <= data.shape[1] or len(np.unique(target)) < 2:
        raise ValueError('Need at least four samples, two classes and a valid feature count')
    transformer = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(score_func=partial(mutual_info_classif, random_state=seed), k=k)),
    ])
    return transformer.fit(data, target)

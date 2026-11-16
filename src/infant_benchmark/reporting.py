"""Read-only validation for separately generated reports; never fits models."""
from __future__ import annotations

import numpy as np
from infant_benchmark.audit import LABELS
from infant_benchmark.metrics import evaluate_predictions


def validate_fold(records, split, rows):
    """Check fold membership, hash isolation and saved prediction semantics."""
    n = len(records)
    train = [int(i) for i in split['train']]
    heldout = [int(i) for i in split['heldout']]
    if len(train) != len(set(train)) or len(heldout) != len(set(heldout)):
        raise ValueError('Duplicate split indices')
    if set(train) & set(heldout) or set(train) | set(heldout) != set(range(n)):
        raise ValueError('Train/heldout must form an exact partition')
    if {records[i]['sha256'] for i in train} & {records[i]['sha256'] for i in heldout}:
        raise ValueError('Identical hash appears in train and heldout')
    seen, labels, probs = set(), [], []
    for row in rows:
        index = int(row['record_index'])
        if index not in heldout or index in seen:
            raise ValueError('Invalid or duplicate prediction index')
        seen.add(index)
        if row['true_label'] != records[index]['label'] or row['true_label'] not in LABELS:
            raise ValueError('Prediction label differs from record metadata')
        vector = np.asarray([float(row[f'prob_{label}']) for label in LABELS])
        if not np.isfinite(vector).all() or (vector < 0).any() or (vector > 1).any() or not np.isclose(vector.sum(), 1, atol=1e-5):
            raise ValueError('Invalid probability vector')
        if row['predicted_label'] != LABELS[int(np.argmax(vector))]:
            raise ValueError('Prediction label differs from maximum probability')
        labels.append(LABELS.index(row['true_label']))
        probs.append(vector)
    if seen != set(heldout):
        raise ValueError('Held-out predictions incomplete')
    return {'metrics': evaluate_predictions(np.asarray(labels), np.asarray(probs)),
            'indices': sorted(seen), 'labels': labels, 'probabilities': probs}

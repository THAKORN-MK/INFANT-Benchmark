import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_fscore_support
from infant_benchmark.audit import LABELS


def evaluate_predictions(labels, probabilities):
    y, p = np.asarray(labels), np.asarray(probabilities)
    if y.ndim != 1 or p.shape != (len(y), 5) or not np.isfinite(p).all():
        raise ValueError('Expected labels and finite five-class probabilities')
    predicted = np.argmax(p, axis=1)
    result = {'accuracy': float(accuracy_score(y, predicted)),
              'balanced_accuracy': float(balanced_accuracy_score(y, predicted)),
              'macro_f1': float(f1_score(y, predicted, labels=list(range(5)), average='macro', zero_division=0)),
              'confusion_matrix': confusion_matrix(y, predicted, labels=list(range(5))).tolist()}
    precision, recall, f1, support = precision_recall_fscore_support(y, predicted, labels=list(range(5)), zero_division=0)
    result['per_class'] = {name: {'precision': float(precision[i]), 'recall': float(recall[i]),
                                  'f1': float(f1[i]), 'support': int(support[i])}
                           for i, name in enumerate(LABELS)}
    result['macro_auc_ovr'] = float(roc_auc_score(y, p, multi_class='ovr', average='macro')) if len(np.unique(y)) == 5 else None
    return result

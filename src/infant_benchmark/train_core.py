"""Mathematical training components; epsilon and weighting are approximations."""
import torch
import torch.nn.functional as F


def class_weights(labels, *, classes=5):
    values = torch.as_tensor(labels, dtype=torch.long)
    if values.ndim != 1 or not values.numel() or torch.any(values < 0) or torch.any(values >= classes):
        raise ValueError('Expected non-empty labels in class range')
    counts = torch.bincount(values, minlength=classes).float()
    if torch.any(counts == 0):
        raise ValueError('Training fold must contain all classes')
    return values.numel() / (classes * counts)


def poly_loss(logits, labels, weights, *, epsilon=1.0):
    ce = F.cross_entropy(logits, labels, weight=weights, label_smoothing=0.05)
    py = torch.softmax(logits, dim=1).gather(1, labels[:, None]).squeeze(1)
    return ce + epsilon * ((1 - py) ** 2).mean()

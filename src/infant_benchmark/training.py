"""Explicit, approximate 5-fold execution; called only on user request."""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import uuid

import joblib
import numpy as np
import torch
from sklearn import __version__ as sklearn_version
from torch.utils.data import DataLoader, TensorDataset

from infant_benchmark.audio import load_waveform
from infant_benchmark.audit import LABELS
from infant_benchmark.features import acoustic_features
from infant_benchmark.fold_features import fit_feature_transform
from infant_benchmark.metrics import evaluate_predictions
from infant_benchmark.manifest import inspect_manifest
from infant_benchmark.model import SincMambaKAN
from infant_benchmark.records import load_records, suffix_disagreements
from infant_benchmark.splits import plan_folds
from infant_benchmark.train_core import class_weights, poly_loss
from infant_benchmark.training_safety import check_output_location


def _write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def train_cv(args):
    root = args.dataset_root.resolve()
    if not root.is_dir():
        raise ValueError(f'Dataset directory does not exist: {root}')
    check_output_location(root, args.output_root)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable in this PyTorch environment; no training started')
    # Fail before reading data or creating outputs if the real Mamba dependency is absent.
    try:
        import mamba_ssm
        from mamba_ssm import Mamba  # noqa: F401
    except (ImportError, OSError) as exc:
        raise RuntimeError('mamba_ssm or its CUDA backend unavailable; no training started') from exc
    try:
        probe = SincMambaKAN().to('cuda').eval()
        test_logits = probe(torch.randn((1, 32000), device='cuda'), torch.randn((1, 100), device='cuda'))
        if tuple(test_logits.shape) != (1, 5) or not torch.isfinite(test_logits).all():
            raise RuntimeError('unexpected or non-finite synthetic output')
        # Synthetic gradient check only: no optimizer, no weight update and no real audio.
        test_logits.square().mean().backward()
        gradients = [p.grad for p in probe.parameters() if p.requires_grad]
        if any(g is None or not torch.isfinite(g).all() for g in gradients):
            raise RuntimeError('missing or non-finite synthetic gradient')
        del probe, test_logits
    except Exception as exc:
        raise RuntimeError(f'Synthetic GPU forward/gradient check failed; no training started: {exc}') from exc
    audit = inspect_manifest(root)
    if audit['missing_count'] or audit['errors'] or audit['label_folder_mismatches']:
        raise ValueError('Dataset metadata/files failed integrity checks; no training started')
    records = load_records(root, args.tier)
    suffix_issues = suffix_disagreements(records)
    print(f"Dataset audit: {audit['cross_tier_hash_groups']} cross-tier duplicate groups; "
          f"{audit['cross_label_hash_groups']} duplicate groups with conflicting labels across the dataset; "
          f"{len(suffix_issues)} selected-tier filename/label mismatches. "
          "Only the selected tier is used; metadata/parent folder is authoritative; no relabelling.", flush=True)
    labels = np.array([r['target'] for r in records], dtype=np.int64)
    groups = [r['sha256'] for r in records]
    folds = plan_folds(labels, groups, seed=args.seed)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:10]
    run = args.output_root.resolve() / run_id
    run.mkdir(parents=True, exist_ok=False)
    config = {'run_id': run_id, 'tier': args.tier, 'seed': args.seed, 'epochs': args.epochs,
              'batch_size': args.batch_size, 'dataset_root': str(root), 'labels': LABELS,
              'protocol': '5-fold stratified exact-byte-hash groups, fixed epochs, outer held-out evaluated once',
              'metric_unit': 'each clip is counted, including duplicates; not independent infant estimates',
              'identity': 'not verified; not infant-disjoint',
              'feature_schema': '213 proxy, not verified against author',
              'deviations': ['Gaussian RBF KAN proxy; padding/crop and filter initialisation assumptions',
                             'noise std 0.005 waveform / 0.01 standardized feature; circular shift <=1600 samples',
                             'PolyLoss epsilon 1.0; fixed epochs; exact-byte grouping',
                             'seed 42 by default; no selection on held-out folds'],
              'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                           'numpy': np.__version__, 'sklearn': sklearn_version,
                           'mamba_ssm': getattr(mamba_ssm, '__version__', 'unknown'), 'cuda': torch.version.cuda,
                           'librosa': version('librosa'), 'scipy': version('scipy'), 'soundfile': version('soundfile')},
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(Path(__file__).parent.glob('*.py'))},
              'gpu': torch.cuda.get_device_name(0)}
    _write_json(run / 'config.json', config)
    _write_json(run / 'records.json', [{k: str(v) if isinstance(v, Path) else v for k, v in r.items()} for r in records])
    _write_json(run / 'data_audit.json', {'full_dataset': audit, 'selected_tier_suffix_disagreements': suffix_issues,
                                          'policy': 'one tier only; metadata and parent folder labels authoritative; group exact duplicates in folds; no relabelling'})
    print(f'Run: {run}; loading {len(records)} waveforms and proxy features', flush=True)
    waves, descriptors = [], []
    for index, record in enumerate(records):
        x = load_waveform(record['path'])
        feats, names = acoustic_features(x)
        waves.append(x)
        descriptors.append(feats)
        if (index + 1) % 100 == 0 or index + 1 == len(records):
            print(f'Extracted {index + 1}/{len(records)}', flush=True)
    waves, descriptors = np.stack(waves), np.stack(descriptors)
    _write_json(run / 'feature_schema.json', {'source': 'documented proxy, not author implementation', 'names': names})
    results = []
    device = torch.device('cuda')
    for fold_number, (train_idx, heldout_idx) in enumerate(folds, start=1):
        fold = run / f'fold_{fold_number}'
        fold.mkdir(exist_ok=False)
        _write_json(fold / 'split.json', {'train': train_idx, 'heldout': heldout_idx})
        transform = fit_feature_transform(descriptors[train_idx], labels[train_idx], k=100, seed=args.seed)
        joblib.dump(transform, fold / 'feature_transform.joblib')
        features_train = np.asarray(transform.transform(descriptors[train_idx]), dtype=np.float32)
        features_test = np.asarray(transform.transform(descriptors[heldout_idx]), dtype=np.float32)
        weights = class_weights(labels[train_idx]).to(device)
        torch.manual_seed(args.seed + fold_number)
        torch.cuda.manual_seed_all(args.seed + fold_number)
        model = SincMambaKAN().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        dataset = TensorDataset(torch.from_numpy(waves[train_idx]), torch.from_numpy(features_train), torch.from_numpy(labels[train_idx]))
        generator = torch.Generator().manual_seed(args.seed + fold_number)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
        with (fold / 'history.csv').open('x', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['epoch', 'train_loss'])
            for epoch in range(args.epochs):
                model.train()
                total = 0.0
                for x, f, y in loader:
                    x, f, y = x.to(device), f.to(device), y.to(device)
                    # These magnitudes are assumptions; no augmentation is used in held-out data.
                    shifts = torch.randint(-1600, 1601, (x.shape[0],), device=device)
                    x = torch.stack([torch.roll(row, int(shift.item())) for row, shift in zip(x, shifts)])
                    x = x + 0.005 * torch.randn_like(x)
                    f = f + 0.01 * torch.randn_like(f)
                    optimizer.zero_grad(set_to_none=True)
                    loss = poly_loss(model(x, f), y, weights, epsilon=1.0)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f'Non-finite loss at fold {fold_number}, epoch {epoch + 1}')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    optimizer.step()
                    total += float(loss.detach()) * len(y)
                scheduler.step()
                mean_loss = total / len(train_idx)
                writer.writerow([epoch + 1, mean_loss])
                stream.flush()
                print(f'Fold {fold_number}/5 Epoch {epoch + 1}/{args.epochs} train_loss={mean_loss:.5f}', flush=True)
        torch.save(model.state_dict(), fold / 'model.pt')
        model.eval()
        probabilities = []
        eval_set = TensorDataset(torch.from_numpy(waves[heldout_idx]), torch.from_numpy(features_test))
        with torch.no_grad():
            for x, f in DataLoader(eval_set, batch_size=args.batch_size):
                probabilities.append(torch.softmax(model(x.to(device), f.to(device)), dim=1).cpu().numpy())
        probabilities = np.concatenate(probabilities)
        metrics = evaluate_predictions(labels[heldout_idx], probabilities)
        _write_json(fold / 'metrics.json', metrics)
        with (fold / 'predictions.csv').open('x', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['record_index', 'true_label', 'predicted_label', *[f'prob_{x}' for x in LABELS]])
            for index, probs in zip(heldout_idx, probabilities):
                writer.writerow([index, LABELS[labels[index]], LABELS[int(np.argmax(probs))], *[float(p) for p in probs]])
        results.append(metrics)
    summary = {'run_id': run_id, 'run_dir': str(run), 'scope': config['protocol'], 'folds': len(results),
               'metrics': {key: {'mean': float(np.mean([r[key] for r in results])),
                                 'std': float(np.std([r[key] for r in results]))}
                           for key in ('accuracy', 'balanced_accuracy', 'macro_f1', 'macro_auc_ovr')}}
    _write_json(run / 'summary.json', summary)
    return summary

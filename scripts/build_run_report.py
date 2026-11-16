"""Validate existing predictions and create figures, without model fitting.

Writes only to a new output directory; never modifies the original run.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from infant_benchmark.audit import LABELS
from infant_benchmark.manifest import resolve_metadata_path
from infant_benchmark.metrics import evaluate_predictions
from infant_benchmark.reporting import validate_fold


def read_json(path):
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def analyse(run, dataset_root, source_root):
    config = read_json(run / 'config.json')
    summary = read_json(run / 'summary.json')
    records = read_json(run / 'records.json')
    if summary['run_id'] != config['run_id'] or run.name != config['run_id']:
        raise ValueError('Run identity mismatch')
    if config['labels'] != list(LABELS) or summary['folds'] != 5:
        raise ValueError('Unexpected label order or fold count')
    audio_mismatch = []
    for i, row in enumerate(records):
        path = resolve_metadata_path(dataset_root, row['relative_path'])
        if not path.is_file() or digest(path) != row['sha256']:
            audio_mismatch.append(i)
    if audio_mismatch:
        raise ValueError(f'Audio missing/changed for record indices: {audio_mismatch[:20]}')
    source_checks = {name: (source_root / name).is_file() and digest(source_root / name) == expected
                     for name, expected in config['source_sha256'].items()}
    values, seen, all_y, all_p, histories = [], [], [], [], []
    for number in range(1, 6):
        folder = run / f'fold_{number}'
        for filename in ('model.pt', 'feature_transform.joblib', 'history.csv', 'metrics.json', 'predictions.csv', 'split.json'):
            if not (folder / filename).is_file() or (folder / filename).stat().st_size == 0:
                raise ValueError(f'Absent/empty artifact: {folder.name}/{filename}')
        fold = validate_fold(records, read_json(folder / 'split.json'), read_csv(folder / 'predictions.csv'))
        saved = read_json(folder / 'metrics.json')
        for key in ('accuracy', 'balanced_accuracy', 'macro_f1', 'macro_auc_ovr'):
            if not np.isclose(fold['metrics'][key], saved[key], atol=1e-9, rtol=1e-7):
                raise ValueError(f'Metric disagreement: fold {number}, {key}')
        if fold['metrics']['confusion_matrix'] != saved['confusion_matrix']:
            raise ValueError(f'Confusion matrix disagreement: fold {number}')
        history = read_csv(folder / 'history.csv')
        if [int(x['epoch']) for x in history] != list(range(1, int(config['epochs']) + 1)):
            raise ValueError(f'Incomplete or unordered epoch history in fold {number}')
        losses = np.asarray([float(x['train_loss']) for x in history])
        if not np.isfinite(losses).all():
            raise ValueError(f'Non-finite training loss in fold {number}')
        histories.append(losses)
        values.append(fold['metrics'])
        seen.extend(fold['indices'])
        all_y.extend(fold['labels'])
        all_p.extend(fold['probabilities'])
    if Counter(seen) != Counter(range(len(records))):
        raise ValueError('Five-fold held-out coverage is not exactly once')
    for key in ('accuracy', 'balanced_accuracy', 'macro_f1', 'macro_auc_ovr'):
        observed = np.asarray([m[key] for m in values])
        if not np.isclose(observed.mean(), summary['metrics'][key]['mean'], atol=1e-9, rtol=1e-7):
            raise ValueError(f'Summary mean disagreement: {key}')
        if not np.isclose(observed.std(), summary['metrics'][key]['std'], atol=1e-9, rtol=1e-7):
            raise ValueError(f'Summary standard deviation disagreement: {key}')
    pooled = evaluate_predictions(np.asarray(all_y), np.asarray(all_p))
    return {'run_id': config['run_id'], 'records': len(records), 'source_hashes_match': source_checks,
            'audio_hashes_match_count': len(records) - len(audio_mismatch),
            'coverage': len(seen), 'fold_metrics': values, 'pooled_metrics': pooled,
            'histories': [h.tolist() for h in histories], 'limitations': [
                'Checkpoint and transform have no original independent integrity manifest; file existence only checked.',
                'Predictions and scores were not regenerated from model weights; no training or model inference was run.',
                'Exact byte grouping is not subject independence; repeated clips contribute to reported counts.',
            ]}


def save_figures(analysis, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matrix = np.asarray(analysis['pooled_metrics']['confusion_matrix'])
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 130})
    for kind, data, name in [('count', matrix, 'confusion_matrix_counts.png'),
                             ('percent', matrix / matrix.sum(axis=1, keepdims=True) * 100, 'confusion_matrix_row_percent.png')]:
        fig, ax = plt.subplots(figsize=(8, 6.8))
        plot = ax.imshow(data, cmap='Blues', vmin=0, vmax=100 if kind == 'percent' else None)
        ax.set(xticks=range(5), yticks=range(5), xticklabels=LABELS, yticklabels=LABELS,
               xlabel='Predicted label', ylabel='True label', title=f'Pooled 5-fold confusion matrix ({kind})')
        for i in range(5):
            for j in range(5):
                val = data[i, j]
                ax.text(j, i, f'{val:.1f}%' if kind == 'percent' else str(int(val)),
                        ha='center', va='center', color='white' if val > data.max() * .55 else '#17334c')
        fig.colorbar(plot, ax=ax, label='Percent of true class' if kind == 'percent' else 'Clips')
        fig.tight_layout()
        fig.savefig(directory / name, bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, losses in enumerate(analysis['histories'], 1):
        ax.plot(range(1, len(losses) + 1), losses, label=f'Fold {i}', linewidth=1.3)
    ax.set(title='Training loss by fold (no validation loss saved)', xlabel='Epoch', ylabel='Training loss')
    ax.grid(alpha=.25)
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / 'training_loss_by_fold.png', bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(LABELS)
    x = np.arange(5)
    per_class = analysis['pooled_metrics']['per_class']
    for shift, metric in [(-.25, 'precision'), (0, 'recall'), (.25, 'f1')]:
        ax.bar(x + shift, [100 * per_class[name][metric] for name in labels], .24, label=metric.capitalize())
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 105), ylabel='Percent', title='Pooled 5-fold performance by class')
    ax.grid(axis='y', alpha=.2)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(directory / 'per_class_metrics.png', bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if output == run or output.is_relative_to(run) or output == args.dataset_root.resolve() or output.is_relative_to(args.dataset_root.resolve()):
        raise ValueError('Report directory must be outside the run and dataset')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    print('Validating predictions, splits, metrics, source hashes and audio hashes...', flush=True)
    result = analyse(run, args.dataset_root.resolve(), args.source_root.resolve())
    output.mkdir(parents=True, exist_ok=False)
    save_figures(result, output)
    with (output / 'verification.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(f'Report figures and verification saved: {output}', flush=True)
    print(json.dumps({'records': result['records'], 'audio_hashes_match_count': result['audio_hashes_match_count'],
                      'source_hashes_all_match': all(result['source_hashes_match'].values()),
                      'pooled_metrics': result['pooled_metrics']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

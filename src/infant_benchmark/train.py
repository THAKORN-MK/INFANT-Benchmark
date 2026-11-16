"""5-fold paper-inspired benchmark. Never runs without explicit --execute."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--tier', choices=('Dunstan-Core', 'InfantCry-1214'), default='InfantCry-1214')
    p.add_argument('--output-root', type=Path, default=Path('runs'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--acknowledge-approximation', action='store_true')
    p.add_argument('--acknowledge-data-issues', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not args.execute:
        print('Dry run only: no data loaded, no model trained, no files written. Execution requires --execute, --acknowledge-approximation, and --acknowledge-data-issues.')
        return 0
    if not args.acknowledge_approximation:
        print('error: acknowledge approximate implementation with --acknowledge-approximation', file=sys.stderr)
        return 2
    if not args.acknowledge_data_issues:
        print('error: review dataset findings and acknowledge with --acknowledge-data-issues', file=sys.stderr)
        return 2
    if args.epochs < 1 or args.batch_size < 1:
        print('error: epochs and batch size must be positive', file=sys.stderr)
        return 2
    try:
        from infant_benchmark.training import train_cv
        result = train_cv(args)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_json(path: Path):
    try:
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def main():
    p = argparse.ArgumentParser(description='Summarize implant experiment runs.')
    p.add_argument('--runs-root', default='/workspace/implant_outputs')
    p.add_argument('--out', default='/workspace/implant_outputs/experiment_summary.csv')
    args = p.parse_args()
    root = Path(args.runs_root)
    rows = []
    for run in sorted([p for p in root.glob('*') if p.is_dir()]):
        row = {'run_dir': str(run), 'run_name': run.name}
        a = load_json(run / 'args.json')
        row.update({f'arg_{k}': v for k, v in a.items() if isinstance(v, (str, int, float, bool, type(None)))})
        for metric_file in ['best_valid_metrics.json', 'test_metrics.json', 'valid_metrics.json', 'test_prototype_metrics.json', 'best_valid_prototype_metrics.json']:
            m = load_json(run / 'metrics' / metric_file)
            for k in ['accuracy', 'macro_f1', 'weighted_f1', 'macro_precision', 'macro_recall']:
                if k in m:
                    row[f'{metric_file.replace(".json","")}_{k}'] = m[k]
        hist = run / 'metrics' / 'history.csv'
        if hist.exists():
            try:
                df = pd.read_csv(hist)
                if 'valid_macro_f1' in df.columns:
                    best = df.sort_values('valid_macro_f1', ascending=False).iloc[0]
                    row['history_best_epoch'] = int(best['epoch'])
                    row['history_best_valid_macro_f1'] = float(best['valid_macro_f1'])
                    if 'valid_accuracy' in best:
                        row['history_best_valid_accuracy'] = float(best['valid_accuracy'])
            except Exception:
                pass
        rows.append(row)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print('SUMMARY_CSV=', out)
    print(pd.DataFrame(rows).tail(20).to_string(index=False) if rows else 'No runs found.')


if __name__ == '__main__':
    main()

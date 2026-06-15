from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]) -> Dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(class_names)))
    out = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)),
        'macro_precision': float(precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
    }
    report = classification_report(y_true, y_pred, labels=labels, target_names=list(class_names), zero_division=0, output_dict=True)
    out['classification_report'] = report
    return out


def save_metrics_json(path: str | Path, metrics: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def save_confusion_matrix(path_csv: str | Path, path_png: str | Path, y_true, y_pred, class_names: Sequence[str]) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    path_csv = Path(path_csv)
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(path_csv)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(max(6, len(class_names) * 0.9), max(5, len(class_names) * 0.8)))
        ax = fig.add_subplot(111)
        im = ax.imshow(cm)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion matrix')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path_png = Path(path_png)
        path_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path_png, dpi=180)
        plt.close(fig)
    except Exception as e:
        print('Warning: could not save confusion matrix image:', repr(e))


def save_predictions(path_csv: str | Path, rows: List[Dict]) -> None:
    path = Path(path_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)

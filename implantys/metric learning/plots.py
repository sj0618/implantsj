#!/usr/bin/env python3
"""
시각화 — t-SNE / 학습곡선 / 혼동행렬 / 클래스별 F1 / 종횡비 진단.

  python plots.py --all                 # 추출된 조건 전부, 그릴 수 있는 것 다
  python plots.py --tsne M1 M4
  python plots.py --confusion M1
  python plots.py --curves
  python plots.py --per-class
  python plots.py --aspect M1

모든 그림은 figures/ 에 저장되고 report.py 가 md 에 임베드한다.

주의: t-SNE 는 거리를 보존하지 않는다. 뭉쳐 보인다고 성능이 좋은 게 아니고,
정량 지표(macro-F1)를 뒷받침하는 용도로만 쓴다.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dataset import load_split  # noqa: E402
from metrics import distance_matrix  # noqa: E402
from prototype import EMB_ROOT, build_prototypes, load_embeddings  # noqa: E402
from run_cells import SCENARIOS, scenarios_for  # noqa: E402

_ROOT = Path(__file__).parent
FIG_DIR = _ROOT / "figures"
LOG_DIR = _ROOT / "logs"
PERCLASS_JSON = _ROOT / "results_per_class.json"

TSNE_PERPLEXITY = 20  # test 195장 기준. 표본이 적어 기본값 30 은 과하다.

# 한글 폰트가 없는 환경에서 라벨이 깨지는 것만 막는다 (제목은 영문 병기).
matplotlib.rcParams["axes.unicode_minus"] = False


def _colors(n):
    base = plt.get_cmap("tab20")
    return [base(i % 20) for i in range(n)]


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path.relative_to(_ROOT)}")
    return path


def _scenario_classes(meta, cond):
    """조건별 (시나리오코드, 평가 클래스 리스트). run_cells.py 규칙을 그대로 쓴다."""
    out = []
    for scen in scenarios_for(cond):
        scope = SCENARIOS[scen][0]
        out.append((scen, sorted(meta["classes"] if scope == "all"
                                 else meta["open_classes"])))
    return out


def _predict(cond, classes, metric, split="test"):
    tr_v, tr_l, _ = load_embeddings(cond, "train")
    te_v, te_l, te_p = load_embeddings(cond, split)
    keep_tr = np.isin(tr_l, classes)
    keep_te = np.isin(te_l, classes)
    protos = build_prototypes(tr_v[keep_tr], tr_l[keep_tr], classes, metric)
    dist = distance_matrix(te_v[keep_te], protos, metric)
    idx = {c: i for i, c in enumerate(classes)}
    y_true = np.array([idx[l] for l in te_l[keep_te]])
    return y_true, dist.argmin(1), te_p[keep_te]


# ---------------------------------------------------------------- t-SNE


def plot_tsne(cond, split="test", out_dir=FIG_DIR, seed=42):
    from sklearn.manifold import TSNE

    meta = load_split()
    vecs, labels, _ = load_embeddings(cond, split)
    classes = sorted(meta["classes"])
    closed = set(meta["closed_classes"])

    perp = min(TSNE_PERPLEXITY, max(5, (len(vecs) - 1) // 3))
    xy = TSNE(n_components=2, perplexity=perp, init="pca",
              random_state=seed, max_iter=1000).fit_transform(vecs)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    cols = _colors(len(classes))
    for i, c in enumerate(classes):
        m = labels == c
        if not m.any():
            continue
        is_closed = c in closed
        ax.scatter(xy[m, 0], xy[m, 1], s=44 if is_closed else 26,
                   color=cols[i], alpha=0.85,
                   marker="^" if is_closed else "o",
                   edgecolors="black" if is_closed else "none",
                   linewidths=0.9 if is_closed else 0,
                   label=f"{c} ({int(m.sum())}){' *' if is_closed else ''}")

    ax.set_title(f"t-SNE — {cond} / {split}\n"
                 f"triangle + black edge = closed (small classes)", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, Path(out_dir) / f"tsne_{cond}_{split}.png")


# ---------------------------------------------------------------- 혼동행렬


def plot_confusion(cond, metric="cosine", out_dir=FIG_DIR, normalize=True):
    """조건의 모든 시나리오에 대해 혼동행렬. 행=정답, 열=예측.

    행 정규화(recall 기준)가 기본이다. 불균형이 53:1 이라 원시 개수로 보면
    adin/bicon 행만 진하게 나와서 소수 클래스가 어디로 새는지 안 보인다.
    """
    meta = load_split()
    closed = set(meta["closed_classes"])
    out = []
    for scen, classes in _scenario_classes(meta, cond):
        y_true, y_pred, _ = _predict(cond, classes, metric)
        n = len(classes)
        cm = np.zeros((n, n), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

        show = cm.astype(float)
        if normalize:
            row = cm.sum(1, keepdims=True)
            show = np.divide(show, np.maximum(row, 1))

        fig, ax = plt.subplots(figsize=(max(6.5, 0.62 * n), max(5.5, 0.55 * n)))
        im = ax.imshow(show, cmap="Blues", vmin=0, vmax=1 if normalize else None)
        fig.colorbar(im, ax=ax, fraction=0.046,
                     label="recall" if normalize else "count")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels([f"{c}*" if c in closed else c for c in classes], fontsize=8)
        for i, c in enumerate(classes):  # closed 클래스 라벨을 빨갛게
            if c in closed:
                ax.get_yticklabels()[i].set_color("crimson")

        thr = show.max() * 0.55 if show.max() else 0.5
        for i in range(n):
            for j in range(n):
                if cm[i, j] == 0:
                    continue
                txt = f"{show[i,j]:.2f}" if normalize else str(cm[i, j])
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color="white" if show[i, j] > thr else "black")

        acc = float((y_true == y_pred).mean())
        ax.set_xlabel("predicted")
        ax.set_ylabel("true  (* = closed, not trained in E3)")
        ax.set_title(f"Confusion — {cond} / {scen} / {metric}\n"
                     f"row-normalized (recall), Top-1 = {acc:.4f}", fontsize=10)
        fig.tight_layout()
        out.append(_save(fig, Path(out_dir) / f"confusion_{cond}_{scen}_{metric}.png"))
    return out


# ---------------------------------------------------------------- 클래스별 F1


def plot_per_class_f1(per_class_json=PERCLASS_JSON, out_dir=FIG_DIR, metric="cosine"):
    """클래스별 F1 을 조건별로 겹쳐 그린다. 소수 클래스가 어디서 무너지는지 본다."""
    p = Path(per_class_json)
    if not p.is_file():
        print(f"{p} 없음. run_cells.py 를 먼저 실행할 것.")
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    meta = load_split()
    closed = set(meta["closed_classes"])
    counts = meta["counts"]

    keys = [k for k in data if k.endswith(f"|{metric}")]
    if not keys:
        print(f"{metric} 결과가 없다.")
        return None
    # 13개 클래스를 다 담은 시나리오만 (E1/E3/BASE13) — 비교 축을 맞춘다
    keys = [k for k in keys if len(data[k]) == len(meta["classes"])]
    if not keys:
        print("13개 클래스 시나리오 결과가 아직 없다.")
        return None

    classes = sorted(meta["classes"], key=lambda c: -counts[c])
    x = np.arange(len(classes))
    w = min(0.8 / len(keys), 0.25)

    fig, ax = plt.subplots(figsize=(max(9, 0.85 * len(classes)), 5))
    for i, k in enumerate(sorted(keys)):
        f1 = [data[k].get(c, {}).get("f1", 0.0) for c in classes]
        cond, scen, _ = k.split("|")
        ax.bar(x + (i - (len(keys) - 1) / 2) * w, f1, w, label=f"{cond} {scen}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={counts[c]})" for c in classes], fontsize=8)
    for i, c in enumerate(classes):
        if c in closed:
            ax.get_xticklabels()[i].set_color("crimson")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Per-class F1 ({metric}) — sorted by sample count\n"
                 f"red label = closed (small classes)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, Path(out_dir) / f"per_class_f1_{metric}.png")


# ---------------------------------------------------------------- 학습곡선


def plot_curves(log_dir=LOG_DIR, out_dir=FIG_DIR):
    """train.py 가 남긴 logs/{cond}.jsonl 을 읽어 곡선을 그린다."""
    log_dir = Path(log_dir)
    logs = sorted(log_dir.glob("*.jsonl")) if log_dir.is_dir() else []
    if not logs:
        print(f"{log_dir} 에 학습 로그가 없다. 학습 조건(M2~M7)을 먼저 돌릴 것.")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for p in logs:
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if not rows:
            continue
        ep = [r["epoch"] for r in rows]
        cond = p.stem
        # CE 끼리 비교한다. SupCon 조건의 train_loss 에는 SupCon 항이 섞여 있어
        # val_loss(CE 전용)와 같은 축에 놓으면 오해를 부른다.
        ce = [r.get("train_ce", r["train_loss"]) for r in rows]
        axes[0].plot(ep, ce, label=f"{cond} train CE")
        axes[0].plot(ep, [r["val_loss"] for r in rows], "--", label=f"{cond} val CE")
        axes[1].plot(ep, [r["val_macro_f1"] for r in rows], label=cond)
        axes[2].plot(ep, [v - c for v, c in zip((r["val_loss"] for r in rows), ce)],
                     label=cond)

    axes[0].set_title("cross-entropy — solid train / dashed val")
    axes[1].set_title("val macro-F1  (model selection criterion)")
    axes[2].set_title("val CE − train CE  (overfit gap)")
    axes[2].axhline(0, color="gray", lw=0.8)
    for a in axes:
        a.set_xlabel("epoch")
        a.legend(fontsize=7)
        a.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, Path(out_dir) / "learning_curves.png")


# ---------------------------------------------------------------- 종횡비 진단


def plot_aspect(cond, metric="cosine", out_dir=FIG_DIR,
                inventory=_ROOT / "inventory.csv"):
    """종횡비 구간별 Top-1 — 왜곡 224 리사이즈가 실제로 해가 되는지 확인.

    실측 종횡비가 0.20 ~ 4.23 (21배)이라, 극단 구간에서만 정확도가 떨어진다면
    그건 리사이즈 방식 문제이지 임베딩 품질 문제가 아니다.
    """
    if not Path(inventory).is_file():
        print(f"{inventory} 없음. make_split.py 를 먼저 실행할 것.")
        return None
    ar = {}
    with open(inventory, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["aspect_ratio"]:
                ar[Path(r["path"]).name] = float(r["aspect_ratio"])

    meta = load_split()
    classes = sorted(meta["classes"])
    y_true, y_pred, paths = _predict(cond, classes, metric)
    correct = (y_true == y_pred)
    ratios = np.array([ar.get(Path(p).name, np.nan) for p in paths])
    ok = ~np.isnan(ratios)
    correct, ratios = correct[ok], ratios[ok]

    edges = [0, 0.4, 0.6, 0.8, 1.25, 2.0, 10.0]
    names, accs, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ratios >= lo) & (ratios < hi)
        if m.sum() == 0:
            continue
        names.append(f"{lo:g}-{hi:g}")
        accs.append(float(correct[m].mean()))
        ns.append(int(m.sum()))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, accs, color="steelblue")
    for b, n in zip(bars, ns):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f"n={n}", ha="center", fontsize=9)
    ax.axhline(float(correct.mean()), color="crimson", ls="--", lw=1,
               label=f"overall Top-1 = {correct.mean():.3f}")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("aspect ratio (w/h) — 1.0 is square")
    ax.set_ylabel("Top-1")
    ax.set_title(f"Accuracy by aspect ratio — {cond} / {metric}\n"
                 f"if only extreme bins drop, distortion resize is a suspect")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for n, a, c in zip(names, accs, ns):
        print(f"    종횡비 {n:<10} n={c:>4}  Top-1={a:.4f}")
    return _save(fig, Path(out_dir) / f"aspect_{cond}_{metric}.png")


# ---------------------------------------------------------------- CLI


def available_conds(emb_root=EMB_ROOT):
    root = Path(emb_root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir()
                  if (p / "train.npz").is_file() and (p / "test.npz").is_file())


def run_all(metric="cosine"):
    conds = available_conds()
    if not conds:
        print("추출된 임베딩이 없다. extract.py 를 먼저 실행할 것.")
        return
    print(f"조건: {conds}")
    print("\n[학습곡선]")
    plot_curves()
    print("\n[클래스별 F1]")
    plot_per_class_f1(metric=metric)
    for c in conds:
        print(f"\n[{c}] t-SNE / 혼동행렬 / 종횡비")
        plot_tsne(c)
        plot_confusion(c, metric)
        plot_aspect(c, metric)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="그릴 수 있는 것 전부")
    ap.add_argument("--tsne", nargs="*", default=None)
    ap.add_argument("--confusion", nargs="*", default=None)
    ap.add_argument("--aspect", nargs="*", default=None)
    ap.add_argument("--curves", action="store_true")
    ap.add_argument("--per-class", action="store_true")
    ap.add_argument("--metric", default="cosine", choices=("l2", "cosine"))
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    if args.all:
        run_all(args.metric)
        return

    did = False
    if args.tsne is not None:
        for c in args.tsne:
            plot_tsne(c, args.split)
        did = True
    if args.confusion is not None:
        for c in args.confusion:
            plot_confusion(c, args.metric)
        did = True
    if args.aspect is not None:
        for c in args.aspect:
            plot_aspect(c, args.metric)
        did = True
    if args.curves:
        plot_curves()
        did = True
    if args.per_class:
        plot_per_class_f1(metric=args.metric)
        did = True
    if not did:
        ap.print_help()


if __name__ == "__main__":
    main()

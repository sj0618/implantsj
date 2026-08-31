#!/usr/bin/env python3
"""
LOO / 거부 실험 시각화.

  python plots_loo.py

산출물 (result_figures/ 또는 figures/)
  loo_unseen_compare.png     조건별 미학습 성능 — 기존 13장 vs LOO 86장
  loo_per_class_recall.png   미학습 4개 클래스별 recall
  reject_score_dist.png      거부 점수 분포 (known vs unknown) + AUROC

읽는 파일: results_loo.csv, results_loo_per_class.json,
          results_loo_scores.json, results.csv, split.json
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_ROOT = Path(__file__).parent
matplotlib.rcParams["axes.unicode_minus"] = False


def _figdir():
    for n in ("result_figures", "figures"):
        if (_ROOT / n).is_dir():
            return _ROOT / n
    p = _ROOT / "result_figures"
    p.mkdir(exist_ok=True)
    return p


FIG = _figdir()
LABEL = {"M1": "M1 Base\n(no training)", "M3": "M3 Head\n(backbone frozen)",
         "M5": "M5 FullFT", "M5b": "M5b FullFT\n(rerun)", "M7": "M7 SupCon"}


def _save(fig, name):
    p = FIG / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {p.relative_to(_ROOT)}")
    return p


def load(name):
    p = _ROOT / name
    if not p.is_file():
        raise SystemExit(f"{name} 없음. loo_eval.py 를 먼저 실행할 것.")
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    return list(csv.DictReader(open(p, newline="", encoding="utf-8")))


# ---------------------------------------------------------------- 1


def plot_unseen_compare(metric="cosine"):
    """기존 13장 평가 vs LOO 86장 평가. 표본을 늘리면 무엇이 달라지는가."""
    loo = {r["cond"]: r for r in load("results_loo.csv") if r["metric"] == metric}
    old = {}
    for r in load("results.csv"):
        if r["scenario"] == "E3" and r["metric"] == metric and r["unseen_macro_f1"]:
            old[r["cond"]] = float(r["unseen_macro_f1"])

    conds = [c for c in ("M1", "M3", "M5", "M5b", "M7") if c in loo]
    x = np.arange(len(conds))
    w = 0.38

    old_v = [old.get(c, np.nan) for c in conds]
    new_v = [float(loo[c]["unseen_macro_f1"]) for c in conds]
    n_new = int(loo[conds[0]]["n_closed_loo"])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, old_v, w, label="old eval (13 test imgs)", color="#bbbbbb")
    ax.bar(x + w / 2, new_v, w, label=f"LOO eval ({n_new} imgs)", color="#2b6cb0")
    for i, v in enumerate(new_v):
        ax.text(i + w / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=9)
    for i, v in enumerate(old_v):
        if not np.isnan(v):
            ax.text(i - w / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=9,
                    color="#666666")

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL.get(c, c) for c in conds], fontsize=9)
    ax.set_ylabel("macro-F1 on 4 unseen brands")
    ax.set_ylim(0, max(new_v) * 1.35)
    ax.set_title(f"Classifying UNSEEN brands ({metric})\n"
                 f"eval samples 13 -> {n_new}", fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, f"loo_unseen_compare_{metric}.png")


# ---------------------------------------------------------------- 2


def plot_per_class_recall(metric="cosine"):
    """미학습 4개를 클래스별로. 어느 브랜드가 먼저 무너지는가."""
    pc = load("results_loo_per_class.json")
    meta = json.loads((_ROOT / "split.json").read_text(encoding="utf-8"))
    closed = sorted(meta["closed_classes"], key=lambda c: -meta["counts"][c])
    conds = [c for c in ("M1", "M3", "M5", "M5b", "M7") if f"{c}|{metric}" in pc]

    x = np.arange(len(closed))
    w = min(0.8 / len(conds), 0.2)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, c in enumerate(conds):
        v = [pc[f"{c}|{metric}"][cls]["recall"] for cls in closed]
        ax.bar(x + (i - (len(conds) - 1) / 2) * w, v, w,
               label=LABEL.get(c, c).replace("\n", " "))

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={meta['counts'][c]})" for c in closed])
    ax.set_ylabel("recall")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Per-class recall on unseen brands ({metric})\n"
                 f"smaller brands collapse first after fine-tuning", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, f"loo_per_class_recall_{metric}.png")


# ---------------------------------------------------------------- 3


def plot_reject(metric="cosine"):
    """거부 점수 분포. 아는 것과 모르는 것이 갈리는가."""
    sc = load("results_loo_scores.json")
    rows = {r["cond"]: r for r in load("results_loo.csv") if r["metric"] == metric}
    conds = [c for c in ("M1", "M3", "M5", "M7") if f"{c}|{metric}" in sc]

    fig, axes = plt.subplots(1, len(conds), figsize=(4.2 * len(conds), 4),
                             sharey=True)
    if len(conds) == 1:
        axes = [axes]
    for ax, c in zip(axes, conds):
        k = np.array(sc[f"{c}|{metric}"]["known"])
        u = np.array(sc[f"{c}|{metric}"]["unknown"])
        bins = np.linspace(min(k.min(), u.min()), max(k.max(), u.max()), 30)
        ax.hist(k, bins=bins, alpha=0.6, label=f"known 9 ({len(k)})",
                color="#2b6cb0", density=True)
        ax.hist(u, bins=bins, alpha=0.6, label=f"unknown 4 ({len(u)})",
                color="#c53030", density=True)
        a = float(rows[c]["reject_auroc"])
        ax.set_title(f"{LABEL.get(c, c)}\nAUROC = {a:.3f}", fontsize=10)
        ax.set_xlabel("dist to nearest known prototype")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("density")
    fig.suptitle(f"Can it tell UNKNOWN from KNOWN? — rejection score ({metric})\n"
                 f"more overlap = worse (AUROC 0.5 = random)", fontsize=12)
    fig.tight_layout()
    return _save(fig, f"reject_score_dist_{metric}.png")




# ---------------------------------------------------------------- 4. 트레이드오프


def plot_tradeoff(metric="cosine"):
    """두 능력의 관계. 추세선은 그리지 않는다 — 관계가 단조롭지 않다.

    M1 → M3(가벼운 학습)은 두 축 모두 좋아지고,
    M3 → M5(backbone 해제)에서만 진짜 맞교환이 일어난다.
    선형 추세선을 그으면 M3 가 이상치처럼 보여 사실을 왜곡한다.
    대신 파레토 경계(어느 축도 손해 없이 이길 수 없는 조건)를 표시한다.
    """
    rows = {r["cond"]: r for r in load("results_loo.csv") if r["metric"] == metric}
    conds = [c for c in ("M1", "M3", "M5", "M5b", "M7") if c in rows]
    pt = {c: (float(rows[c]["unseen_macro_f1"]), float(rows[c]["reject_auroc"]))
          for c in conds}
    col = {"M1": "#718096", "M3": "#2b6cb0", "M5": "#c53030",
           "M5b": "#e07a5f", "M7": "#805ad5"}
    off = {"M1": (0, 18), "M3": (0, 18), "M5": (-8, -26),
           "M5b": (0, 18), "M7": (10, -26)}

    # 파레토 경계: 다른 조건에게 두 축 모두 밀리지 않는 조건
    pareto = [c for c in conds
              if not any(pt[o][0] >= pt[c][0] and pt[o][1] >= pt[c][1] and o != c
                         for o in conds)]

    fig, ax = plt.subplots(figsize=(8, 6.4))
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.text(min(v[0] for v in pt.values()), 0.508, "random rejection",
            fontsize=8, color="gray")

    pf = sorted((pt[c] for c in pareto))
    if len(pf) > 1:
        ax.plot([p[0] for p in pf], [p[1] for p in pf], "-", color="#cbd5e0",
                lw=2.5, zorder=1, label="Pareto frontier")

    for c in conds:
        xi, yi = pt[c]
        edge = "black" if c in pareto else "white"
        ax.scatter(xi, yi, s=240, color=col.get(c, "#333"), zorder=3,
                   edgecolors=edge, linewidths=2)
        ax.annotate(LABEL.get(c, c).replace("\n", " "), (xi, yi),
                    textcoords="offset points", xytext=off.get(c, (0, 18)),
                    ha="center", fontsize=9)

    # M3 → M5 화살표: 맞교환이 일어나는 유일한 구간
    if "M3" in pt and "M5" in pt:
        ax.annotate("", xy=pt["M5"], xytext=pt["M3"],
                    arrowprops=dict(arrowstyle="->", color="#c53030", lw=1.6,
                                    connectionstyle="arc3,rad=-0.15"), zorder=2)
        mx = (pt["M3"][0] + pt["M5"][0]) / 2
        my = (pt["M3"][1] + pt["M5"][1]) / 2
        ax.text(mx, my - 0.03, "unfreeze backbone\n= the only real trade-off",
                fontsize=8.5, color="#c53030", ha="center")

    ax.set_xlabel("classify unseen brands  (macro-F1)  →  better")
    ax.set_ylabel("detect 'I don't know'  (rejection AUROC)  →  better")
    ax.set_title("Two different abilities, and they do not improve together\n"
                 f"black-edged points = Pareto frontier  ({metric})", fontsize=12)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, f"tradeoff_{metric}.png")


# ---------------------------------------------------------------- 5. t-SNE + unknown


def plot_tsne_unknown(conds=("M1", "M5"), seed=42):
    """알려진 9개(색) 위에 미학습 86장(검은 X)을 얹는다.

    거부 AUROC 가 왜 그렇게 나왔는지가 여기서 보인다 —
    검은 X 가 색 군집 안에 섞여 있으면 구별 불가, 바깥에 있으면 구별 가능.
    """
    from sklearn.manifold import TSNE
    from loo_eval import gather
    from dataset import load_split

    meta = load_split()
    open_c = sorted(meta["open_classes"])
    fig, axes = plt.subplots(1, len(conds), figsize=(7.2 * len(conds), 6.4))
    if len(conds) == 1:
        axes = [axes]
    rej = {r["cond"]: r for r in load("results_loo.csv") if r["metric"] == "cosine"}

    for ax, cond in zip(axes, conds):
        _, (ote_v, ote_l), (cl_v, cl_l) = gather(cond, meta)
        X = np.concatenate([ote_v, cl_v])
        perp = min(20, max(5, (len(X) - 1) // 3))
        xy = TSNE(n_components=2, perplexity=perp, init="pca",
                  random_state=seed, max_iter=1000).fit_transform(X)
        k, u = xy[:len(ote_l)], xy[len(ote_l):]

        cmap = plt.get_cmap("tab10")
        for i, c in enumerate(open_c):
            m = ote_l == c
            ax.scatter(k[m, 0], k[m, 1], s=30, color=cmap(i % 10), alpha=0.75,
                       label=c if ax is axes[0] else None)
        ax.scatter(u[:, 0], u[:, 1], s=70, marker="x", c="black", linewidths=1.6,
                   label="UNKNOWN (4 brands)" if ax is axes[0] else None)
        a = float(rej[cond]["reject_auroc"]) if cond in rej else float("nan")
        ax.set_title(f"{LABEL.get(cond, cond)}\nrejection AUROC = {a:.3f}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.02), ncol=5, fontsize=8,
                   frameon=False)
    fig.suptitle("Where do UNKNOWN brands land?  black X = 4 brands never trained on",
                 fontsize=13)
    fig.tight_layout()
    return _save(fig, f"tsne_unknown_{'_'.join(conds)}.png")


if __name__ == "__main__":
    print("LOO / 거부 시각화")
    plot_unseen_compare("cosine")
    plot_per_class_recall("cosine")
    plot_reject("cosine")
    plot_reject("l2")
    plot_tradeoff("cosine")
    plot_tsne_unknown(("M1", "M5"))
    plot_tsne_unknown(("M3", "M7"))

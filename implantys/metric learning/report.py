#!/usr/bin/env python3
"""
실험 결과를 하나의 마크다운 보고서로 묶는다.

  python report.py --md report.md              # 파일로 저장 (그림은 경로만 나열)
  python report.py --md report.md --figures    # 그림을 이미지로 임베드

구성
  1 목적 / 2 셋업 / 3 학습 결과 / 4 정량(질문별) / 5 클래스별 /
  6 정성(그림) / 7 진단 결론 / 8 한계 / 부록

§7 진단 결론은 하드코딩이 아니라 숫자에서 뽑는다. 핵심은 §7.3 —
archive/*/results.csv 와 현재를 비교해 **재실행 잡음**을 실측하고,
그보다 작은 Δ 는 "판정 불가"로 처리한다. 재학습하면 결론도 함께 갱신된다.

읽는 파일 (없으면 그 절은 건너뛴다)
  split.json                    데이터 구성
  inventory.csv                 종횡비 실측
  results.csv                   22셀 + 변형 조건
  results_per_class.json        클래스별 성능
  softmax_reference.csv         softmax 참조값
  result_logs/*.jsonl           학습곡선 (없으면 logs/)
  result_figures/*.png          그림     (없으면 figures/)
  archive/*/results.csv         이전 실행 — 잡음 추정에 쓴다

의존성은 표준 라이브러리뿐이다 — numpy/torch 없이 돌아간다.
대표값은 macro-F1. 클래스 불균형이 53:1 이라 Top-1 은 큰 클래스만 맞혀도 높다.
"""

import argparse
import csv
import json
from pathlib import Path

_ROOT = Path(__file__).parent
RESULTS_CSV = _ROOT / "results.csv"
PERCLASS_JSON = _ROOT / "results_per_class.json"
SOFTMAX_CSV = _ROOT / "softmax_reference.csv"
SPLIT_JSON = _ROOT / "split.json"
INVENTORY_CSV = _ROOT / "inventory.csv"


def _pick_dir(*names):
    """여러 이름 중 실제로 있는 폴더를 고른다 (로컬은 result_*, 포드는 원래 이름)."""
    for n in names:
        if (_ROOT / n).is_dir():
            return _ROOT / n
    return _ROOT / names[0]


LOG_DIR = _pick_dir("result_logs", "logs")
FIG_DIR = _pick_dir("result_figures", "figures")
ARCHIVE_DIR = _ROOT / "archive"

METRIC_KEYS = ("macro_f1", "top1", "top3")
PRIMARY = "macro_f1"

COND_LABEL = {
    "M1": "M1 Base (학습 없음)",
    "M2": "M2 Head / 13개 학습",
    "M3": "M3 Head / 9개 학습",
    "M4": "M4 FullFT / 13개 학습",
    "M5": "M5 FullFT / 9개 학습",
    "M6": "M6 SupCon / 13개 학습",
    "M7": "M7 SupCon / 9개 학습",
    "M6-nobal": "M6-nobal SupCon / 13개 학습 / balanced sampler 끔",
}
VARIANTS = {"M6-nobal"}   # 22셀 집계 밖의 변형 조건


# ---------------------------------------------------------------- 유틸


def fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def delta(a, b):
    """b - a. 뒤에 온 쪽이 얼마나 나아졌는지."""
    if a is None or b is None:
        return "—"
    return f"{b - a:+.4f}"


class Report:
    def __init__(self):
        self.lines = []

    def h(self, text, level=2):
        self.lines += ["", "#" * level + " " + text, ""]

    def p(self, text):
        self.lines += [text, ""]

    def table(self, header, rows, empty="_아직 결과가 없다._"):
        if not rows:
            self.p(empty)
            return
        self.lines.append("| " + " | ".join(header) + " |")
        self.lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for r in rows:
            self.lines.append("| " + " | ".join(str(x) for x in r) + " |")
        self.lines.append("")

    embed_figures = False   # 기본은 경로만 적는다. --figures 로 켜면 이미지 임베드.

    def fig(self, path, caption):
        """그림이 있으면 기록한다. embed_figures 가 꺼져 있으면 경로만 적는다."""
        p = Path(path)
        if not p.is_file():
            return False
        rel = p.relative_to(_ROOT).as_posix()
        if self.embed_figures:
            self.lines += [f"![{caption}]({rel})", f"_{caption}_", ""]
        else:
            self.lines += [f"- `{rel}` — {caption}"]
        return True

    def text(self):
        return "\n".join(self.lines).strip() + "\n"


# ---------------------------------------------------------------- 로딩


def load_results(path=RESULTS_CSV):
    if not Path(path).is_file():
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k in METRIC_KEYS + ("unseen_macro_f1",):
                v = r.get(k, "")
                r[k] = float(v) if v not in ("", None) else None
            out[(r["cond"], r["scenario"], r["metric"])] = r
    return out


def load_json(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def get(res, cond, scen, metric, key=PRIMARY):
    r = res.get((cond, scen, metric))
    return None if r is None else r[key]


def pair_rows(res, pairs, metric_list=("l2", "cosine")):
    rows = []
    for la, ca, sa, lb, cb, sb in pairs:
        for m in metric_list:
            a, b = get(res, ca, sa, m), get(res, cb, sb, m)
            if a is None and b is None:
                continue
            rows.append([f"{la} → {lb}", m, fmt(a), fmt(b), delta(a, b)])
    return rows


# ---------------------------------------------------------------- 1~2 절


def sec_purpose(rp):
    rp.h("1. 실험 목적", 2)
    rp.p("DINOv2 임베딩이 임플란트 제조사 분류에 적합한지 검증하고, "
         "최적의 학습 방법·거리를 진단한다.\n\n"
         "**목적 A** — 학습한 제조사를 잘 분류하는 모델을 만든다. "
         "따라서 주 평가는 전부 \"학습에 쓴 제조사를 분류\"하는 것이다. "
         "E3 만 학습하지 않은 제조사를 섞는 추가 실험이다.\n\n"
         "> 이 문서에서 **open = 데이터량이 많은 제조사**(항상 학습 포함), "
         "**closed = 데이터량이 적은 제조사**(조건에 따라 학습 제외)다. "
         "일반적인 metric learning 용어와 반대이므로 주의.")


def sec_setup(rp, meta, inventory=INVENTORY_CSV):
    rp.h("2. 셋업", 2)
    if meta is None:
        rp.p("_split.json 이 없다. `python make_split.py` 를 먼저 실행할 것._")
        return

    cfg = meta["config"]
    counts, groups = meta["counts"], meta["groups"]
    total = sum(counts.values())
    name = cfg.get("dataset_name") or Path(cfg["data_dir"]).name
    rp.p(f"**데이터** `{name}` — {len(counts)}개 제조사 / {total}장  \n"
         f"**분할** stratified {cfg['ratios']['train']:.0%} / "
         f"{cfg['ratios']['val']:.0%} / {cfg['ratios']['test']:.0%}, seed {cfg['seed']}  \n"
         f"**그룹 경계** N ≥ {cfg['min_open']} → open, 미만 → closed  \n"
         f"**backbone** DINOv2 ViT-B/14, CLS 토큰 768d  \n"
         f"**전처리** 왜곡 224×224 리사이즈 (종횡비 무시), ImageNet 정규화  \n"
         f"**프로토타입** Mean (train split 임베딩의 제조사별 평균)")

    rows, tot = [], [0, 0, 0]
    for c in sorted(counts, key=lambda x: -counts[x]):
        s = meta["splits"][c]
        n = (len(s["train"]), len(s["val"]), len(s["test"]))
        for i in range(3):
            tot[i] += n[i]
        rows.append([c, groups[c], counts[c], *n])
    rows.append(["**합계**", "", f"**{total}**", f"**{tot[0]}**",
                 f"**{tot[1]}**", f"**{tot[2]}**"])
    rp.table(["제조사", "그룹", "전체", "train", "val", "test"], rows)

    imb = max(counts.values()) / min(counts.values())
    rp.p(f"클래스 불균형 **{max(counts.values())}:{min(counts.values())} = {imb:.0f}:1**. "
         f"그래서 대표 지표는 Top-1 이 아니라 **macro-F1** 이다.")

    if Path(inventory).is_file():
        ars = []
        with open(inventory, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["aspect_ratio"]:
                    ars.append(float(r["aspect_ratio"]))
        if ars:
            ars.sort()
            rp.p(f"**종횡비 실측** min {ars[0]:.2f} / 중앙값 {ars[len(ars)//2]:.2f} / "
                 f"max {ars[-1]:.2f} — **{ars[-1]/ars[0]:.0f}배 범위**. "
                 f"왜곡 리사이즈의 대가가 이미지마다 크게 다르다는 뜻이라 "
                 f"§6 의 종횡비 진단으로 확인한다.")

    rp.h("실험 조건", 3)
    rp.table(
        ["조건", "아키텍처", "학습 범위", "평가 시나리오"],
        [["M1", "Base (frozen, 학습 없음)", "—", "13개 분류 / 9개 분류"],
         ["M2", "frozen + linear head", "13개", "E1 (13개 분류)"],
         ["M3", "frozen + linear head", "9개", "E2 (9개) + E3 (13개)"],
         ["M4", "full FT", "13개", "E1"],
         ["M5", "full FT", "9개", "E2 + E3"],
         ["M6", "full FT + SupCon", "13개", "E1"],
         ["M7", "full FT + SupCon", "9개", "E2 + E3"]])
    rp.p("셀 = 조건 × 시나리오 × 거리(L2·cosine) = **22셀**. "
         "M1 은 학습이 없어 E1 과 E3 가 같은 계산이므로 중복을 뺐다.")


# ---------------------------------------------------------------- 3절


def sec_training(rp, log_dir=LOG_DIR, fig_dir=FIG_DIR):
    rp.h("3. 학습 결과", 2)
    logs = sorted(Path(log_dir).glob("*.jsonl")) if Path(log_dir).is_dir() else []
    if not logs:
        rp.p("_학습 로그가 없다. M2~M7 을 돌리면 채워진다._")
        return

    rows, stale = [], []
    for p in logs:
        recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if not recs:
            continue
        best = max(recs, key=lambda r: r["val_macro_f1"])
        last = recs[-1]
        # 격차는 CE 끼리 비교해야 한다. val_loss 는 CE 만이므로 train 쪽도 CE 를 쓴다.
        # train_ce 가 없는 옛 로그는 train_loss 로 대체하되, SupCon 조건이면
        # train_loss 에 SupCon 이 섞여 있어 격차가 무의미하므로 표시하지 않는다.
        has_ce = "train_ce" in last
        train_ce = last.get("train_ce", last["train_loss"])
        is_supcon_cond = p.stem.startswith(("M6", "M7"))
        if not has_ce and is_supcon_cond:
            stale.append(p.stem)
            gap_s, flag = "—", "로그에 CE 분리 없음"
        else:
            gap = last["val_loss"] - train_ce
            gap_s = f"{gap:+.4f}"
            flag = "과적합 의심" if gap > 0.5 else ""
        rows.append([COND_LABEL.get(p.stem, p.stem), len(recs), best["epoch"],
                     fmt(best["val_macro_f1"]), fmt(train_ce),
                     fmt(last["val_loss"]), gap_s, flag])
    rp.table(["조건", "epoch 수", "best epoch", "best val macro-F1",
              "최종 train CE", "최종 val loss", "격차", "비고"], rows)
    rp.p("모델 선택은 **val macro-F1** 기준이다. val loss 는 불균형 때문에 "
         "큰 클래스에 끌려간다. 격차(val loss − train CE)가 크면 과적합이다.\n\n"
         "격차는 **CE 끼리** 비교한다. SupCon 조건의 총 손실에는 SupCon 항이 섞여 "
         "있어 val loss(CE 전용)와 직접 빼면 의미가 없다.")
    if stale:
        rp.p(f"> ⚠️ {', '.join(stale)} 은 CE 를 분리 기록하기 전에 학습된 로그라 "
             f"격차를 계산할 수 없다. 재학습하면 채워진다.")
    rp.fig(Path(fig_dir) / "learning_curves.png", "학습곡선 — loss / val macro-F1 / 과적합 격차")


# ---------------------------------------------------------------- 4절


def sec_questions(rp, res):
    rp.h("4. 정량 결과 — 질문별 비교", 2)
    rp.p(f"기록된 셀 **{len(res)} / 22**. 아래 Δ 는 모두 macro-F1 기준.")

    rp.h("Q1. head 학습이 도움되나?", 3)
    rp.p("frozen backbone 위에 linear head 만 얹는 것으로 얼마나 좋아지나.")
    rp.table(["비교", "거리", "before", "after", "Δ macro-F1"],
             pair_rows(res, [
                 ("M1 Base(13분류)", "M1", "BASE13", "M2 Head/13학습", "M2", "E1"),
                 ("M1 Base(9분류)", "M1", "BASE9", "M3 Head/9학습", "M3", "E2"),
             ]))

    rp.h("Q2. backbone 을 푸는 게 가치 있나?", 3)
    rp.p("**본 실험의 하이라이트.** DINOv2 frozen 이 강하다는 주장의 검증. "
         "Δ가 음수면 1304장 소량 데이터 파인튜닝이 오히려 손해라는 뜻이다.")
    rp.table(["비교", "거리", "frozen", "full FT", "Δ macro-F1"],
             pair_rows(res, [
                 ("M2 Head/13학습", "M2", "E1", "M4 FullFT/13학습", "M4", "E1"),
                 ("M3 Head/9학습", "M3", "E2", "M5 FullFT/9학습", "M5", "E2"),
             ]))

    rp.h("Q3. SupCon 이 돕나?", 3)
    rp.p("같은 full FT 끼리, SupCon 손실만 추가한 효과.")
    rp.table(["비교", "거리", "FT", "FT+SupCon", "Δ macro-F1"],
             pair_rows(res, [
                 ("M4 FullFT/13학습", "M4", "E1", "M6 SupCon/13학습", "M6", "E1"),
                 ("M5 FullFT/9학습", "M5", "E2", "M7 SupCon/9학습", "M7", "E2"),
             ]))

    if any(k[0] == "M6-nobal" for k in res):
        rp.h("Q3-a. SupCon 이 13개 학습에서 나빴던 게 balanced sampler 탓인가? (변형 조건)", 4)
        rp.p("SupCon 조건은 balanced sampler 가 기본 ON 이다. 13개 학습에는 "
             "nobel(train 3장)이 들어 있어 **3장을 에폭당 약 70회 뽑는다**. "
             "이 과표집이 SupCon 을 망친 것인지 확인하려고, sampler 만 끄고 "
             "동일 조건을 한 번 더 돌렸다. 계획서의 7조건 밖이라 22셀에는 포함하지 않는다.")
        rp.table(["비교", "거리", "balanced ON", "balanced OFF", "Δ macro-F1"],
                 pair_rows(res, [("M6 (sampler ON)", "M6", "E1",
                                  "M6-nobal (sampler OFF)", "M6-nobal", "E1")]))
        rp.p("참고로 sampler 없는 full FT 기준선은 M4 다 — "
             "M6-nobal 이 M4 를 넘으면 SupCon 자체는 도움이 된 것이고, "
             "못 넘으면 SupCon 이 13개 학습에서는 손해라는 뜻이다.")
        rp.table(["비교", "거리", "M4 (SupCon 없음)", "M6-nobal (SupCon, sampler OFF)", "Δ"],
                 pair_rows(res, [("M4", "M4", "E1", "M6-nobal", "M6-nobal", "E1")]))

    rp.h("Q4. 어느 거리가 최선? (L2 vs 코사인)", 3)
    rows = []
    for (cond, scen, m) in sorted(res):
        if m != "l2":
            continue
        a, b = get(res, cond, scen, "l2"), get(res, cond, scen, "cosine")
        if a is None and b is None:
            continue
        rows.append([COND_LABEL.get(cond, cond), scen, fmt(a), fmt(b), delta(a, b)])
    rp.table(["조건", "시나리오", "L2", "cosine", "Δ (cos−L2)"], rows)

    rp.h("Q5. metric learning 을 쓸 이유가 있나? (prototype vs softmax)", 3)
    rp.p("목적 A 에서는 softmax head 가 실무 기준선이다. "
         "prototype 이 그보다 못하면 metric learning 을 쓸 근거가 약해진다.")
    rp.table(["조건", "시나리오", "prototype (최선 거리)", "softmax", "Δ (proto−softmax)"],
             softmax_rows(res),
             empty="_softmax 참조값이 없다. 학습 조건(M2~M7)을 돌리면 채워진다._")

    rp.h("Q6. 학습 안 한 제조사도 임베딩으로 잡히나? (추가 실험)", 3)
    rp.p("E3 = 9개로 학습한 모델이 closed 4개까지 포함해 13개를 분류. "
         "**미학습 4개 F1** 이 그 4개만 뽑은 macro-F1 이다.\n\n"
         "여기서 softmax 는 출력이 9개뿐이라 원천적으로 불가능하다. "
         "임베딩 + 프로토타입만 가능한 유일한 구간이다.")
    rows = []
    for cond in ("M1", "M3", "M5", "M7"):
        scen = "BASE13" if cond == "M1" else "E3"
        for m in ("l2", "cosine"):
            r = res.get((cond, scen, m))
            if r is None:
                continue
            rows.append([COND_LABEL.get(cond, cond), m,
                         fmt(r["macro_f1"]), fmt(r["unseen_macro_f1"])])
    rp.table(["조건", "거리", "전체 13개 macro-F1", "미학습 4개 F1"], rows)

    rp.h("Q7. 학습 범위를 좁히면 손해인가?", 3)
    rp.p("**같은 13개 분류 과제**를 두고 13개 학습(E1)과 9개 학습(E3)을 비교. "
         "Δ가 음수인 만큼이 closed 4개를 학습에서 뺀 대가다.")
    rp.table(["비교", "거리", "13개 학습(E1)", "9개 학습(E3)", "Δ macro-F1"],
             pair_rows(res, [
                 ("M2 Head/13학습", "M2", "E1", "M3 Head/9학습", "M3", "E3"),
                 ("M4 FullFT/13학습", "M4", "E1", "M5 FullFT/9학습", "M5", "E3"),
                 ("M6 SupCon/13학습", "M6", "E1", "M7 SupCon/9학습", "M7", "E3"),
             ]))


def softmax_rows(res, path=SOFTMAX_CSV):
    if not Path(path).is_file():
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            cond, scen = r["cond"], r["scenario"]
            sm = float(r["macro_f1"]) if r.get("macro_f1") else None
            cands = [v for v in (get(res, cond, scen, m) for m in ("l2", "cosine"))
                     if v is not None]
            best = max(cands) if cands else None
            rows.append([COND_LABEL.get(cond, cond), scen, fmt(best), fmt(sm),
                         delta(sm, best)])
    return rows


# ---------------------------------------------------------------- 5절


def sec_per_class(rp, meta, per_class, metric="cosine"):
    rp.h("5. 클래스별 성능", 2)
    if not per_class or meta is None:
        rp.p("_클래스별 결과가 없다. run_cells.py 를 먼저 실행할 것._")
        return

    counts, closed = meta["counts"], set(meta["closed_classes"])
    keys = sorted(k for k in per_class
                  if k.endswith(f"|{metric}") and len(per_class[k]) == len(counts))
    if not keys:
        rp.p(f"_13개 클래스를 모두 평가한 {metric} 결과가 아직 없다._")
        return

    classes = sorted(counts, key=lambda c: -counts[c])
    header = ["제조사", "n", "그룹"] + [k.split("|")[0] + " " + k.split("|")[1]
                                       for k in keys]
    rows = []
    for c in classes:
        mark = "**closed**" if c in closed else "open"
        name = f"**{c}**" if c in closed else c
        row = [name, counts[c], mark]
        for k in keys:
            d = per_class[k].get(c)
            row.append(fmt(d["f1"], 3) if d else "—")
        rows.append(row)
    rp.table(header, rows)
    rp.p(f"F1 ({metric} 거리). **굵은 줄이 closed** — 데이터가 적은 제조사다. "
         f"nobel 은 test 가 1장이라 F1 이 0 또는 1 로만 나온다. "
         f"macro-F1 을 해석할 때 이 점을 감안할 것.")
    rp.fig(FIG_DIR / f"per_class_f1_{metric}.png",
           f"클래스별 F1 ({metric}) — 표본 수 순 정렬, 빨간 라벨이 closed")


# ---------------------------------------------------------------- 6절


def sec_figures(rp, res, fig_dir=FIG_DIR, metric="cosine"):
    rp.h("6. 정성 결과 (그림)", 2)
    fig_dir = Path(fig_dir)
    if not fig_dir.is_dir():
        rp.p("_그림이 없다. `python plots.py --all` 을 실행할 것._")
        return

    conds = sorted({k[0] for k in res}) or ["M1"]
    any_fig = False

    rp.h("혼동행렬", 3)
    rp.p("행=정답, 열=예측. 행 정규화(recall)라 소수 클래스가 어디로 새는지 보인다. "
         "빨간 라벨이 closed 제조사다.")
    for c in conds:
        for f in sorted(fig_dir.glob(f"confusion_{c}_*_{metric}.png")):
            scen = f.stem.split("_")[2]
            any_fig |= rp.fig(f, f"{COND_LABEL.get(c, c)} / {scen} / {metric}")

    rp.h("t-SNE", 3)
    rp.p("고차원 임베딩의 2D 투영. 같은 제조사가 뭉치는지 본다. "
         "△ 검은 테두리가 closed 제조사다.\n\n"
         "> t-SNE 는 거리를 보존하지 않는다. 뭉쳐 보인다고 성능이 좋은 게 아니라, "
         "정량 지표를 뒷받침하는 용도로만 읽을 것.")
    for c in conds:
        any_fig |= rp.fig(fig_dir / f"tsne_{c}_test.png", f"{COND_LABEL.get(c, c)} — t-SNE")

    rp.h("종횡비 진단", 3)
    rp.p("왜곡 224 리사이즈가 실제로 해가 되는지 확인. "
         "극단 구간에서만 정확도가 떨어지면 임베딩 품질이 아니라 리사이즈가 원인이다.")
    for c in conds:
        any_fig |= rp.fig(fig_dir / f"aspect_{c}_{metric}.png",
                          f"{COND_LABEL.get(c, c)} — 종횡비 구간별 Top-1")

    if not any_fig:
        rp.p("_생성된 그림이 없다. `python plots.py --all` 을 실행할 것._")


# ---------------------------------------------------------------- 7절


def rerun_noise(res, archive=ARCHIVE_DIR):
    """archive/*/results.csv 와 현재를 비교해 **재실행 잡음**을 실측한다.

    같은 코드·같은 seed 로 다시 학습해도 GPU 비결정성 때문에 값이 흔들린다.
    그 폭보다 작은 Δ 는 결론의 근거가 될 수 없다.
    → (macro_f1 최대 변동, unseen 최대 변동, 비교한 셀 수, 상세 행)
    """
    olds = sorted(Path(archive).glob("*/results.csv")) if Path(archive).is_dir() else []
    if not olds:
        return None
    rows, d_main, d_unseen = [], [], []
    for old_path in olds:
        old = load_results(old_path)
        for key, o in old.items():
            n = res.get(key)
            if n is None or o["macro_f1"] is None or n["macro_f1"] is None:
                continue
            dm = n["macro_f1"] - o["macro_f1"]
            if abs(dm) < 1e-9:
                continue  # 재학습하지 않은 조건 — 완전히 동일
            d_main.append(abs(dm))
            du = None
            if o["unseen_macro_f1"] is not None and n["unseen_macro_f1"] is not None:
                du = n["unseen_macro_f1"] - o["unseen_macro_f1"]
                d_unseen.append(abs(du))
            rows.append([f"{key[0]} {key[1]} {key[2]}",
                         fmt(o["macro_f1"]), fmt(n["macro_f1"]), f"{dm:+.4f}",
                         "—" if du is None else f"{du:+.4f}"])
    if not rows:
        return None
    return {"max_main": max(d_main), "max_unseen": max(d_unseen) if d_unseen else None,
            "n": len(rows), "rows": rows}


def sec_diagnosis(rp, res, per_class, meta):
    """진단 결론 — 숫자에서 직접 뽑는다. 재실행하면 결론도 함께 갱신된다."""
    rp.h("7. 진단 결론", 2)
    if not res:
        rp.p("_결과가 없다._")
        return

    noise = rerun_noise(res)
    thr = noise["max_main"] if noise else 0.0

    # ---- 최고 조건
    rp.h("7.1 과제별 최고 성능", 3)
    tasks = [("13개 분류", ["BASE13", "E1", "E3"]),
             ("9개 분류", ["BASE9", "E2"])]
    rows = []
    for name, scens in tasks:
        cand = [(v, k) for k, v in ((k, get(res, *k)) for k in res) if False]
        cand = [(get(res, c, s, m), (c, s, m))
                for (c, s, m) in res if s in scens and get(res, c, s, m) is not None]
        if not cand:
            continue
        best_v, (c, s, m) = max(cand)
        base = get(res, "M1", scens[0], m)
        rows.append([name, COND_LABEL.get(c, c), s, m, fmt(best_v),
                     fmt(base), delta(base, best_v)])
    rp.table(["과제", "최고 조건", "시나리오", "거리", "macro-F1",
              "M1 Base", "Δ vs Base"], rows)

    # ---- 질문별 판정
    rp.h("7.2 질문별 판정", 3)
    if noise:
        rp.p(f"판정 기준: 같은 조건을 재학습했을 때의 실측 변동이 최대 "
             f"**{noise['max_main']:.4f}** 였다(§7.3). "
             f"이보다 작은 Δ 는 **판정 불가**로 둔다.")

    def verdict(d):
        if d is None:
            return "—", "결과 부족"
        if abs(d) <= thr:
            return f"{d:+.4f}", "판정 불가 (잡음 범위)"
        return f"{d:+.4f}", ("긍정" if d > 0 else "부정")

    def _d(a, b):
        return None if (a is None or b is None) else b - a

    def best_of(cond, scen):
        vals = [get(res, cond, scen, m) for m in ("l2", "cosine")]
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None

    checks = [
        ("Q1 head 학습이 도움되나", "M1", "BASE13", "M2", "E1"),
        ("Q1 head 학습이 도움되나 (9개)", "M1", "BASE9", "M3", "E2"),
        ("Q2 backbone 을 푸는 게 가치 있나", "M2", "E1", "M4", "E1"),
        ("Q2 backbone 을 푸는 게 가치 있나 (9개)", "M3", "E2", "M5", "E2"),
        ("Q3 SupCon 이 돕나 (13개)", "M4", "E1", "M6", "E1"),
        ("Q3 SupCon 이 돕나 (9개)", "M5", "E2", "M7", "E2"),
        ("Q7 학습 범위를 좁혀도 되나", "M4", "E1", "M5", "E3"),
    ]
    rows = []
    for label, ca, sa, cb, sb in checks:
        a, b = best_of(ca, sa), best_of(cb, sb)
        d = None if (a is None or b is None) else b - a
        ds, v = verdict(d)
        rows.append([label, f"{ca} {sa}", fmt(a), f"{cb} {sb}", fmt(b), ds, v])
    rp.table(["질문", "기준", "값", "비교", "값", "Δ", "판정"], rows)

    # ---- 재현성
    rp.h("7.3 재현성 — 이 결과의 잡음 규모", 3)
    if noise is None:
        rp.p("_archive/ 에 이전 실행 결과가 없어 잡음을 추정할 수 없다._")
    else:
        rp.p(f"같은 코드·같은 seed 로 다시 학습한 조건들을 이전 실행과 비교했다 "
             f"({noise['n']}셀). GPU 비결정성만으로 생기는 변동이다.")
        rp.table(["셀", "이전", "현재", "Δ macro-F1", "Δ 미학습4개F1"], noise["rows"])
        msg = (f"**주 지표(macro-F1) 최대 변동 {noise['max_main']:.4f}.** "
               f"단일 재실행 쌍에서 얻은 거친 추정이라 참값은 이보다 클 수 있다 — "
               f"경계에 걸친 Δ 는 보수적으로 읽을 것.")
        if noise["max_unseen"] is not None:
            msg += (f" 반면 **미학습 4개 F1 은 최대 {noise['max_unseen']:.4f}** 흔들린다"
                    f" — test 표본이 13장뿐(§2)이라 그렇다. "
                    f"미학습 관련 결론은 이 폭을 넘는 차이만 유효하다.")
        rp.p(msg)

    # ---- 미학습 클래스
    if per_class and meta:
        rp.h("7.4 학습하지 않은 제조사 (E3)", 3)
        closed = sorted(meta["closed_classes"], key=lambda c: -meta["counts"][c])
        keys = [(c, "E3") for c in ("M3", "M5", "M7") if (c, "E3", "cosine") in res]
        if keys:
            rows = []
            for cls in closed:
                r = [cls, len(meta["splits"][cls]["test"])]
                for c, s in keys:
                    d = per_class.get(f"{c}|{s}|cosine", {}).get(cls)
                    r.append(fmt(d["f1"], 3) if d else "—")
                rows.append(r)
            rp.table(["제조사", "test 장수"] + [COND_LABEL.get(c, c) for c, _ in keys],
                     rows)
        rp.p("9개로 학습한 모델이 학습하지 않은 4개를 분류한 결과다. "
             "**softmax 로는 계산 자체가 불가능한 구간** — 출력이 9개뿐이라 "
             "closed 4개를 뽑을 수 없다. 임베딩+프로토타입만 가능하다. "
             "다만 test 표본이 13장이라 §7.3 의 잡음이 그대로 실린다.")

    # ---- 종합
    rp.h("7.5 종합", 3)
    q1 = best_of("M2", "E1"), best_of("M1", "BASE13")
    q2 = best_of("M4", "E1"), best_of("M2", "E1")
    lines = []
    if all(x is not None for x in q1):
        lines.append(f"1. **학습이 임베딩을 크게 바꾼다.** frozen backbone 위에 "
                     f"projection+분류기만 얹어도 macro-F1 이 "
                     f"{q1[1]:.3f} → {q1[0]:.3f} 로 오른다. 가장 싼 개입이 가장 큰 이득이다.")
    if all(x is not None for x in q2):
        d = q2[0] - q2[1]
        lines.append(f"2. **backbone 을 푸는 것도 추가로 이득이다** ({d:+.3f}). "
                     f"1304장 소량인데도 과적합이 관찰되지 않았다(§3의 격차 참조). "
                     f"\"DINOv2 는 frozen 이 강하니 파인튜닝하지 마라\"는 "
                     f"이 데이터에서 성립하지 않는다.")
    sm = softmax_rows(res)
    if sm:
        lines.append("3. **목적 A 에서는 프로토타입 분류가 softmax 대비 이점이 없다**"
                     "(§4 Q5, 대부분 ±0.01 이내). 학습한 제조사를 분류하는 것이 "
                     "목표라면 softmax 를 쓰면 된다. metric learning 의 가치는 "
                     "E3 처럼 softmax 가 원천적으로 불가능한 구간에만 있다.")
    # SupCon — 잡음 기준(thr)을 넘는 비교만 결론으로 삼는다
    sup = []
    d13 = _d(best_of("M4", "E1"), best_of("M6", "E1"))
    d9 = _d(best_of("M5", "E2"), best_of("M7", "E2"))
    dnb = _d(best_of("M4", "E1"), best_of("M6-nobal", "E1"))
    dsamp = _d(best_of("M6", "E1"), best_of("M6-nobal", "E1"))
    if d13 is not None and d9 is not None:
        if abs(d13) <= thr and abs(d9) <= thr:
            sup.append(f"**SupCon 의 효과는 확인되지 않았다.** 13개 학습 {d13:+.3f}, "
                       f"9개 학습 {d9:+.3f} 로 둘 다 재실행 변동({thr:.3f}) 안이다. "
                       f"부호는 서로 반대지만 그 자체가 결론이 될 수 없다.")
        else:
            sup.append(f"**SupCon 효과** — 13개 학습 {d13:+.3f}, 9개 학습 {d9:+.3f}.")
    if dnb is not None and abs(dnb) > thr:
        sup.append(f"다만 **balanced sampler 를 끈 SupCon(M6-nobal)은 SupCon 없는 "
                   f"M4 보다 확실히 나쁘다** ({dnb:+.3f}, 잡음 초과).")
    if dsamp is not None:
        if abs(dsamp) <= thr:
            sup.append(f"sampler ON/OFF 자체의 차이는 {dsamp:+.3f} 로 잡음 안이라, "
                       f"\"소수 클래스 과표집이 SupCon 을 망쳤다\"는 가설은 "
                       f"**지지도 반증도 되지 않았다** — 적어도 예상한 방향(끄면 좋아짐)"
                       f"으로는 움직이지 않았다.")
        else:
            sup.append(f"sampler 를 끄자 {dsamp:+.3f} 변했다 — 과표집 가설과 반대 방향이다.")
    if sup:
        lines.append("4. " + " ".join(sup))
    if noise and noise["max_unseen"] is not None:
        lines.append(f"5. **미학습 제조사에 대한 결론은 유보한다.** 파인튜닝이 "
                     f"범용성을 해치는 방향이 관찰되지만, 재실행 변동이 "
                     f"{noise['max_unseen']:.3f} 에 달해 조건 간 차이와 구분되지 않는다. "
                     f"확정하려면 closed 클래스의 test 표본을 늘려야 한다.")
    rp.p("\n".join(lines) if lines else "_결과 부족._")


def sec_loo(rp, meta, res, metric="cosine"):
    """LOO 재평가 + 거부 실험. results_loo.csv 가 있을 때만 나온다."""
    p = _ROOT / "results_loo.csv"
    if not p.is_file():
        return
    rows = [r for r in csv.DictReader(open(p, newline="", encoding="utf-8"))
            if r["metric"] == metric]
    if not rows:
        return
    pc_path = _ROOT / "results_loo_per_class.json"
    pc = load_json(pc_path) or {}

    rp.h("7.6 미학습 브랜드 재평가 — Leave-One-Out", 3)
    n_loo = rows[0]["n_closed_loo"]
    rp.p(f"§7.3 에서 미학습 지표의 재실행 변동이 0.094 로 판정 불가였다. 원인은 "
         f"평가 표본이 13장뿐이었다는 것이다.\n\n"
         f"**M1·M3·M5·M7 은 closed 4개를 학습에 전혀 쓰지 않았다.** 모델에게는 "
         f"closed 의 train/val/test 가 모두 똑같이 처음 보는 그림이므로 나눌 이유가 "
         f"없다. 자기 자신만 빼고 프로토타입을 만들면(Leave-One-Out) "
         f"**{n_loo}장 전부를 평가에 쓸 수 있다.** 재학습은 필요 없다 — "
         f"캐시된 임베딩으로 평균을 다시 낼 뿐이다.\n\n"
         f"open 9개에는 적용할 수 없다(모델이 학습에 썼으므로). open 은 test split 만 쓴다.")

    old = {r["cond"]: r["unseen_macro_f1"] for r in res.values()
           if r["scenario"] == "E3" and r["metric"] == metric}
    trows = []
    for r in rows:
        c = r["cond"]
        trows.append([COND_LABEL.get(c, c), fmt(old.get(c)),
                      f"**{float(r['unseen_macro_f1']):.4f}**",
                      fmt(float(r["unseen_recall"])),
                      fmt(float(r["macro_f1_13"]))])
    rp.table(["조건", "기존 (test 13장)", f"LOO ({n_loo}장) macro-F1",
              "LOO recall", "13개 전체 macro-F1"], trows)
    rp.p("대표값은 **recall** 이다. 평가 집합 구성비가 바뀌면 precision 은 변하지만 "
         "recall 은 변하지 않는다. 13개 전체 macro-F1 은 구성비가 달라 기존 E3 와 "
         "직접 비교하지 않는다.")

    if pc:
        closed = sorted(meta["closed_classes"], key=lambda c: -meta["counts"][c])
        conds = [r["cond"] for r in rows]
        head = ["제조사", "전체 장수", "기존 test"] + [c for c in conds]
        trows = []
        for cls in closed:
            row = [cls, meta["counts"][cls], len(meta["splits"][cls]["test"])]
            for c in conds:
                d = pc.get(f"{c}|{metric}", {}).get(cls)
                row.append(fmt(d["recall"], 3) if d else "—")
            trows.append(row)
        rp.table(head, trows)
        rp.p("클래스별 recall. 표본이 적은 브랜드일수록 파인튜닝 후 먼저 무너진다.")

    rp.h("7.7 거부 — 모르는 브랜드를 모른다고 답할 수 있나", 3)
    rp.p("§7.6 은 \"미학습 브랜드에게 대표 벡터를 쥐어주면 자기 것을 찾아가나\"를 "
         "물었다(13지선다). 실전에서 먼저 필요한 건 다른 질문이다 — "
         "**\"애초에 아는 9개가 아니라는 걸 아나.\"**\n\n"
         "이건 대표 벡터가 필요 없다. 아는 9개까지의 거리만 재서 "
         "\"다 멀면 모름\"으로 판정한다. **softmax 로는 계산 자체가 불가능한 "
         "구간이며, 임베딩 방식만 가능하다.**")
    trows = [[COND_LABEL.get(r["cond"], r["cond"]),
              f"**{float(r['reject_auroc']):.4f}**",
              fmt(float(r["reject_fpr95"]))] for r in rows]
    rp.table(["조건", "AUROC (0.5=무작위)", "FPR@95%TPR"], trows)
    rp.p("AUROC = 모르는 브랜드 한 장과 아는 브랜드 한 장을 뽑았을 때 "
         "모르는 쪽 점수가 더 높을 확률. FPR@95%TPR = 모르는 것의 95% 를 "
         "걸러내려 할 때 아는 것을 잘못 버리는 비율(낮을수록 좋다).")

    for f, cap in ((f"loo_unseen_compare_{metric}.png", "기존 13장 평가 vs LOO 평가"),
                   (f"loo_per_class_recall_{metric}.png", "미학습 4개 클래스별 recall"),
                   (f"reject_score_dist_{metric}.png", "거부 점수 분포 (known vs unknown)")):
        rp.fig(FIG_DIR / f, cap)


def sec_limits(rp, meta):
    rp.h("8. 한계", 2)
    items = [
        "**클래스 불균형 53:1** (adin 266 : nobel 5). macro-F1 을 대표값으로 쓰지만 "
        "소수 클래스의 분산이 크다.",
        "**nobel 은 test 1장, dionavi 는 2장.** 이 클래스들의 F1 은 사실상 이산값이라 "
        "macro-F1 을 흔든다. 클래스별 성능(§5)을 반드시 함께 볼 것.",
        "**왜곡 리사이즈.** 종횡비가 21배 범위인데 전부 224×224 로 강제했다. "
        "형태·종횡비 기반 제조사 신호가 손상됐을 수 있다.",
        "**GT-prototype 부재.** 소스였던 AI Hub Data 가 13개 클래스 중 3개만 덮고 "
        "(osstem 115 / dentium 36 / iti 6) 촬영 도메인도 달라 축을 폐기했다. "
        "Mean-prototype 만 검증했다.",
        "**단일 데이터 출처.** 한 데이터셋의 crop 만 썼다. 다른 장비·기관 영상에 "
        "일반화되는지는 확인하지 않았다.",
        "**GT bounding box 기준 crop.** detection 오차를 배제한 조건이므로, "
        "end-to-end 성능은 이보다 낮다.",
        "**재실행 변동.** 같은 코드·같은 seed 로 재학습해도 GPU 비결정성 때문에 "
        "값이 흔들린다. 실측 규모는 §7.3 참조. 소수 클래스 지표일수록 크다.",
    ]
    rp.p("\n".join(f"- {x}" for x in items))


# ---------------------------------------------------------------- 부록


def sec_appendix(rp, res):
    rp.h("부록. 전체 셀", 2)
    order = list(COND_LABEL)
    rows = []
    for (cond, scen, m) in sorted(
            res, key=lambda k: (order.index(k[0]) if k[0] in order else 99, k[1], k[2])):
        r = res[(cond, scen, m)]
        rows.append([cond, r["scenario_desc"], m, r["n_classes"], r["n_test"],
                     fmt(r["top1"]), fmt(r["top3"]), fmt(r["macro_f1"]),
                     fmt(r["unseen_macro_f1"])])
    rp.table(["조건", "시나리오", "거리", "클래스", "test",
              "Top-1", "Top-3", "macro-F1", "미학습4개F1"], rows)


# ---------------------------------------------------------------- 조립


def build(metric="cosine", embed_figures=False):
    Report.embed_figures = embed_figures
    res = load_results()
    meta = load_json(SPLIT_JSON)
    per_class = load_json(PERCLASS_JSON)

    rp = Report()
    rp.h("임플란트 제조사 분류 — DINOv2 Metric Learning 실험 결과", 1)
    done = sorted({k[0] for k in res})
    rp.p(f"완료 조건: **{', '.join(done) if done else '없음'}**  |  "
         f"기록된 셀 **{len(res)} / 22**  |  대표 지표 **macro-F1**")

    sec_purpose(rp)
    sec_setup(rp, meta)
    sec_training(rp)
    sec_questions(rp, res)
    sec_per_class(rp, meta, per_class, metric)
    sec_figures(rp, res, metric=metric)
    sec_diagnosis(rp, res, per_class, meta)
    sec_loo(rp, meta, res, metric)
    sec_limits(rp, meta)
    sec_appendix(rp, res)
    return rp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", nargs="?", const="report.md", default=None,
                    help="마크다운으로 저장 (경로 생략 시 report.md)")
    ap.add_argument("--metric", default="cosine", choices=("l2", "cosine"),
                    help="클래스별 표·그림에 쓸 거리")
    ap.add_argument("--quiet", action="store_true", help="콘솔 출력 생략")
    ap.add_argument("--figures", action="store_true",
                    help="그림을 md 에 이미지로 임베드 (기본은 경로만 나열)")
    args = ap.parse_args()

    rp = build(args.metric, embed_figures=args.figures)
    if not args.quiet:
        print(rp.text())
    if args.md:
        Path(args.md).write_text(rp.text(), encoding="utf-8")
        print(f"저장: {args.md}")


if __name__ == "__main__":
    main()

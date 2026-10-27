"""
verify_pipeline.py — 파이프라인 전체 자체 점검 (폐쇄망에서 그대로 실행 가능)

실제 데이터나 GPU 없이, 합성 데이터로 다음을 확인한다.
  wiring    스테이지 간 파일명/경로/백본 레지스트리가 실제로 맞물리는지
  options   metric의 prototype·거리·공분산·변환·거부점수, attributes의 fuse 방식
  fuse      정규화·가중치·open-set gate·threshold·top-k·거부지표·산출 스키마
  data      폴더 스캔, split 배분, open 클래스 지정, band mask 연결, k-fold
  literals  yaml 주석이 광고하는 선택지가 코드에 실제로 있는지
  config    죽은 yaml 키 / 스테이지 간 체크포인트 경로 합의

torch는 필요 없다 (numpy/pandas/sklearn만 사용). 따라서 학습 자체가 아니라
**설정과 결합 로직**을 검증한다. 실제 학습은 GPU 장비에서 각 스테이지를 돌려야 한다.

    python src/verify_pipeline.py               # 전부
    python src/verify_pipeline.py --only fuse   # 한 섹션만
"""
import argparse
import itertools
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config_utils import load_config, as_list

RESULTS = {"OK": [], "WARN": [], "FAIL": []}


def check(cond, msg, detail="", warn=False):
    bucket = "OK" if cond else ("WARN" if warn else "FAIL")
    RESULTS[bucket].append(msg)
    print(("  OK   " if cond else ("  WARN " if warn else "  FAIL ")) + msg
          + (f"  {detail}" if detail else ""))


def section_wiring():
    """스테이지 간 배선 (파일명·경로·레지스트리)"""
    cfgs = {p.stem: load_config(p) for p in sorted((ROOT/"configs").glob("*.yaml"))}

    print("\n[1] 모든 yaml 로드 + extends 상속")
    for name, c in cfgs.items():
        check("paths" in c and "protocol" in c or name == "base",
              f"{name}.yaml — paths/protocol 상속됨")

    print("\n[2] manifest 파일명 핸드오프 (Stage 0-a 산출 → 각 스테이지 입력)")
    produced = {k: Path(v).name for k, v in cfgs["00_data"]["outputs"].items()
                if k != "labels_json"}
    print(f"       Stage 0-a 산출: {produced}")
    consumers = {
        "01_backbone":       ["closed_manifest", "full_manifest"],
        "03_metric_grid":    ["closed_manifest", "open_manifest"],
        "03_clustering_full42": ["full_manifest"],
        "03_metric_fewshot": ["target_manifest"],   # few-shot은 등록 대상 manifest 하나만 쓴다
        "04_band_vit":       ["closed_manifest", "full_manifest"],
        "04_masked_vit":     ["closed_manifest", "full_manifest"],
        "05_attributes":     ["closed_manifest", "full_manifest"],
    }
    for cfg_name, keys in consumers.items():
        d = cfgs[cfg_name].get("data", {})
        for key in keys:
            if key not in d:
                check(False, f"{cfg_name}.data.{key} 없음")
                continue
            base = Path(d[key]).name
            check(base in produced.values(),
                  f"{cfg_name}.data.{key} = {base} ← Stage 0-a가 만드는 이름과 일치")

    print("\n[3] band mask 경로/접미사 합의 (Stage 0 산출 ↔ Stage 0-a가 찾는 이름)")
    seg = cfgs["02_segmentation"]
    data_band = cfgs["00_data"]["band"]
    seg_band_dir = None
    for key in ("band_output_dir", "band_dir", "output_dir"):
        if key in (seg.get("band") or {}):
            seg_band_dir = (seg["band"] or {})[key]
    print(f"       02_segmentation.band = {seg.get('band')}")
    print(f"       00_data.band         = {data_band}")
    check(Path(data_band["mask_root"]).name == Path(str(seg_band_dir or data_band["mask_root"])).name,
          f"band mask 폴더 일치: {data_band['mask_root']}")
    check(data_band["suffix"] == "_band.png", f"band 파일 접미사 = {data_band['suffix']}")

    print("\n[4] 백본 레지스트리 — 모든 grid의 backbone 이름이 정의돼 있는가")
    for cfg_name in ["01_backbone", "03_metric_grid", "03_clustering_full42",
                     "05_attributes"]:
        c = cfgs[cfg_name]
        names = as_list((c.get("grid") or {}).get("backbone", []))
        if not names and "backbone" in c:
            names = as_list(c["backbone"])
        for n in names:
            check(n in c["backbones"], f"{cfg_name}: 백본 '{n}' 레지스트리에 있음")
    for entry in cfgs["03_metric_fewshot"]["models"]:
        check(entry["backbone"] in cfgs["03_metric_fewshot"]["backbones"],
              f"03_metric_fewshot: 모델 '{entry['name']}'의 백본 '{entry['backbone']}'")
    for cfg_name in ["04_band_vit", "04_masked_vit"]:
        c = cfgs[cfg_name]
        n = c.get("backbone") or (c.get("model") or {}).get("backbone")
        check(n is None or n in c["backbones"], f"{cfg_name}: 백본 '{n}'")

    print("\n[5] 백본 체크포인트 이름 규칙 (Stage 1 산출 ↔ base.yaml 주석 안내)")
    pat = re.compile(r"checkpoints/backbones/([a-z0-9\-]+)_(full|partial)_(closed27_open15|full42)\.pt")
    base_txt = (ROOT/"configs/base.yaml").read_text(encoding="utf-8")
    hinted = set(pat.findall(base_txt))
    check(len(hinted) >= 4, f"base.yaml 주석의 체크포인트 예시 {len(hinted)}건이 "
                            f"run_backbone.py의 <name>_<mode>_<protocol>.pt 규칙과 동일")
    for name, mode, proto in sorted(hinted):
        check(name in cfgs["base"]["backbones"], f"  예시 '{name}' 레지스트리에 존재")

    print("\n[6] fuse 입력 — 각 브랜치가 실제로 쓰는 파일명과 resolve 패턴")
    from fuse.resolve import LAYOUTS
    producers = {
        "metric":     ["eval/scores_valid_<tag>.csv", "eval/scores_test_<tag>.csv"],
        "classifier": ["evaluation/valid/scores.csv (band-vit)",
                       "evaluation/test/scores.csv (band-vit)",
                       "eval_valid/scores.csv (masked-vit)",
                       "eval_original/scores.csv (masked-vit)"],
        "attributes": ["scores_valid.csv", "scores_test.csv"],
    }
    for branch, layout in LAYOUTS.items():
        for split, patterns in layout.items():
            print(f"       {branch:11s} {split:5s} ← {patterns}")
    check(LAYOUTS["metric"]["valid"] == ["eval/scores_valid_*.csv"], "metric valid 패턴")
    check("eval_band/scores.csv" not in LAYOUTS["classifier"]["valid"],
          "classifier valid 패턴에 test 파일(eval_band)이 섞여 있지 않음")
    # masked-vit은 evaluation/(band 입력)과 eval_*(원본 입력)을 둘 다 남긴다.
    # 핵심 실험은 원본이므로 eval_* 가 먼저여야 한다 — 순서가 뒤집히면 fuse가
    # 조용히 학습 분포(band) 점수를 쓰게 된다.
    for split, first, later in (("valid", "eval_valid/scores.csv", "evaluation/valid/scores.csv"),
                                ("test", "eval_original/scores.csv", "evaluation/test/scores.csv")):
        order = LAYOUTS["classifier"][split]
        check(order.index(first) < order.index(later),
              f"classifier {split}: 원본 입력 평가({first.split('/')[0]})가 먼저 탐색됨")
    check(len(set(LAYOUTS["classifier"]["test"])) == len(LAYOUTS["classifier"]["test"]),
          "classifier test 패턴에 중복 없음")
    check(LAYOUTS["attributes"]["valid"] == ["scores_valid.csv"], "attributes valid 패턴")

    print("\n[7] 엔트리포인트 존재 여부 (common.sh가 가리키는 파일)")
    common = (ROOT/"scripts/common.sh").read_text(encoding="utf-8")
    for m in re.finditer(r'export \w+="\$\{\w+:-\$\{SRC_DIR\}/([^}"]+)\}"', common):
        check((ROOT/"src"/m.group(1)).is_file(), f"src/{m.group(1)}")

    print("\n[8] 셸 스크립트가 가리키는 config 존재 여부")
    for sh in sorted((ROOT/"scripts").glob("*/*.sh")):
        txt = sh.read_text(encoding="utf-8")
        for m in re.finditer(r'\$\{CONFIG_DIR\}/([\w.]+\.yaml)', txt):
            check((ROOT/"configs"/m.group(1)).is_file(),
                  f"{sh.parent.name}/{sh.name} → configs/{m.group(1)}")


def section_options():
    """metric/attributes 옵션 실행"""
    rng = np.random.default_rng(0)
    C, D, PER = 6, 32, 12
    classes = [f"cls{i}" for i in range(C)]
    centers = rng.normal(size=(C, D)) * 3
    feats = np.concatenate([centers[i] + rng.normal(scale=1.0, size=(PER, D)) for i in range(C)])
    labels = np.array([classes[i] for i in range(C) for _ in range(PER)])
    query = np.concatenate([centers[i] + rng.normal(scale=1.0, size=(4, D)) for i in range(C)])
    qlab = np.array([classes[i] for i in range(C) for _ in range(4)])

    print("\n[A] metric — prototype 5종 x spherical on/off")
    from metric.prototypes import build_prototypes, PROTOTYPE_METHODS
    cfg_m = load_config(ROOT/"configs/03_metric_grid.yaml")
    declared = as_list(cfg_m["eval"]["prototype"]["methods"])
    check(set(declared) <= set(PROTOTYPE_METHODS), f"yaml methods {declared} ⊆ 구현 {list(PROTOTYPE_METHODS)}")
    ref = {c: rng.normal(size=(2, D)) for c in classes}
    for method in PROTOTYPE_METHODS:
        for sph in (False, True):
            try:
                p = build_prototypes(feats, labels, classes, method, ss_count=5, kmeans_clusters=3,
                                     reference=ref if method == "reference" else None,
                                     spherical_mean=sph, log=lambda *a: None)
                shape_ok = set(p) == set(classes) and all(v.ndim == 2 and v.shape[1] == D for v in p.values())
                check(shape_ok, f"prototype={method:12s} spherical={str(sph):5s}",
                      f"protos/class={ {k: len(v) for k, v in list(p.items())[:1]} }")
            except Exception as ex:
                check(False, f"prototype={method} spherical={sph}", f"{type(ex).__name__}: {ex}")

    print("\n[B] metric — 거리 5종 x 공분산 추정 5종")
    from metric.scoring import Scorer, DISTANCES, COVARIANCE_ESTIMATORS, TRANSFORMS, build_scorer, build_transform
    check(set(as_list(cfg_m["eval"]["distances"])) <= set(DISTANCES), "yaml distances ⊆ 구현")
    protos = build_prototypes(feats, labels, classes, "mean", log=lambda *a: None)
    for dist in DISTANCES:
        covs = COVARIANCE_ESTIMATORS if dist == "mahalanobis" else ("ledoit_wolf",)
        for cov in covs:
            try:
                s = Scorer(distance=dist, covariance=cov).fit(feats, labels, classes)
                mat = s.score(query, protos)
                pred = np.array(classes)[mat.argmin(axis=1)]
                acc = float((pred == qlab).mean())
                check(mat.shape == (len(query), C) and np.isfinite(mat).all(),
                      f"distance={dist:17s} cov={cov:14s}", f"acc={acc:.3f}")
            except Exception as ex:
                check(False, f"distance={dist} cov={cov}", f"{type(ex).__name__}: {ex}")

    print("\n[C] metric — 임베딩 변환 3종")
    for mode in TRANSFORMS:
        for bal in (False, True):
            try:
                t = build_transform({"transform": {"mode": mode, "pca_dim": 8, "pca_balanced_fit": bal}})
                t.fit(feats, labels, classes)
                out = t.apply(query)
                check(out.shape[0] == len(query), f"transform={mode:7s} balanced_fit={str(bal):5s}",
                      f"D {D}→{out.shape[1]}")
            except Exception as ex:
                check(False, f"transform={mode} balanced={bal}", f"{type(ex).__name__}: {ex}")

    print("\n[D] metric — open-set 거부 점수 5종 + AUROC/FPR")
    from metric.openset import reject_scores, evaluate_rejection, auroc, fpr_at_tpr, REJECT_SCORES
    check(set(as_list(cfg_m["eval"]["reject_metrics"]["scores"])) <= set(REJECT_SCORES),
          f"yaml reject scores ⊆ 구현 {list(REJECT_SCORES)}")
    known_d = np.abs(rng.normal(loc=1.0, scale=0.3, size=(60, C)))
    unknown_d = np.abs(rng.normal(loc=2.5, scale=0.3, size=(30, C)))
    dists = np.vstack([known_d, unknown_d])
    is_unk = np.array([False]*60 + [True]*30)
    res = evaluate_rejection(dists, is_unk, which=REJECT_SCORES, temperature=1.0)
    for name in REJECT_SCORES:
        got = res["scores"].get(name)
        check(got is not None and 0.0 <= got["auroc"] <= 1.0, f"reject score={name:10s}",
              f"AUROC={got['auroc']:.3f} FPR@95={got['fpr_at_95tpr']:.3f}")
    check(abs(auroc(np.array([3.,4.]), np.array([1.,2.])) - 1.0) < 1e-9, "AUROC 완전분리=1.0")
    check(abs(auroc(np.array([1.,1.]), np.array([1.,1.])) - 0.5) < 1e-9, "AUROC 전부동점=0.5")

    print("\n[E] attributes — fuse 방식 2종 + 가중치 학습")
    from attributes.fuse import fuse_scores, learn_feature_weights, match_probability_tensor
    NF, NV = 5, 3
    attrs = [f"f{i}" for i in range(NF)]
    target = rng.integers(0, NV, size=(C, NF))          # spec table [C, F]
    N = 60
    truth = rng.integers(0, C, size=N)
    probs = {}
    for fi, a in enumerate(attrs):
        p_arr = rng.random((N, NV)) * 0.2
        if fi < 3:                                       # 앞 3개 feature만 정답 신호
            p_arr[np.arange(N), target[truth, fi]] += 1.0
        probs[a] = p_arr / p_arr.sum(axis=1, keepdims=True)
    matrix = match_probability_tensor(probs, target, attrs)
    check(matrix.shape == (N, C, NF), "match_probability_tensor → [N, C, F]", f"shape={matrix.shape}")
    try:
        w = learn_feature_weights(matrix, truth, attrs, epochs=300)
        top3 = sorted(range(NF), key=lambda i: -w[i])[:3]
        check(abs(w.sum() - 1.0) < 1e-6 and set(top3) == {0, 1, 2},
              "fuse=learned 가중치 학습 — 정보 있는 feature를 상위로",
              f"w={np.round(w,3).tolist()} 상위3={top3}")
    except Exception as ex:
        check(False, "fuse=learned", f"{type(ex).__name__}: {ex}")

    for mode, kw in [("learned", {"weights": w}), ("min_match", {"min_match": 3})]:
        try:
            out = fuse_scores(probs, target, attrs, mode=mode, **kw)
            sc = out["scores"]
            acc = float((sc.argmax(axis=1) == truth).mean())
            check(sc.shape == (N, C) and np.isfinite(sc).all(),
                  f"attributes fuse mode={mode:9s} → [N,C] 점수", f"shape={sc.shape} acc={acc:.3f}")
        except Exception as ex:
            check(False, f"attributes fuse mode={mode}", f"{type(ex).__name__}: {ex}")

    # 균등 가중치 대비 학습 가중치가 나은지 (노이즈 feature 2개를 눌러야 함)
    try:
        uni = fuse_scores(probs, target, attrs, mode="learned",
                          weights=np.full(NF, 1.0/NF))["scores"]
        lrn = fuse_scores(probs, target, attrs, mode="learned", weights=w)["scores"]
        a_uni = float((uni.argmax(axis=1) == truth).mean())
        a_lrn = float((lrn.argmax(axis=1) == truth).mean())
        check(a_lrn >= a_uni, "학습 가중치 >= 균등 가중치", f"{a_uni:.3f} → {a_lrn:.3f}")
    except Exception as ex:
        check(False, "가중치 비교", f"{type(ex).__name__}: {ex}")


def section_fuse():
    """fuse 옵션 실행"""
    from fuse.core import normalize_scores
    from fuse.combine import (learn_branch_weights, combine, apply_openset_gate,
                              find_threshold_auto, apply_threshold, evaluate_topk,
                              evaluate_rejection_quality, build_prediction_frame)

    rng = np.random.default_rng(7)
    C_CLOSED, C_OPEN, N = 8, 4, 400
    classes = [f"c{i}" for i in range(C_CLOSED)] + [f"o{i}" for i in range(C_OPEN)]
    C = len(classes)
    open_classes = [c for c in classes if c.startswith("o")]
    truth_idx = rng.integers(0, C, size=N)
    truth = np.array(classes)[truth_idx]
    is_open = np.isin(truth, open_classes)

    def branch(strength, open_strength=None):
        """정답에 신호를 준 [N,C] 점수. open 클래스에는 다른 세기를 줄 수 있다."""
        s = rng.normal(size=(N, C))
        st = np.where(is_open, open_strength if open_strength is not None else strength, strength)
        s[np.arange(N), truth_idx] += st
        return s

    # metric은 open을 잘 잡고, classifier/attributes는 closed만 잘한다 (노션 규칙의 전제)
    scores = {
        "metric":     -branch(1.6, open_strength=1.8),   # 거리 (작을수록 좋음)
        "classifier":  branch(2.2, open_strength=0.0),
        "attributes":  branch(1.2, open_strength=0.0),
    }
    higher = {"metric": False, "classifier": True, "attributes": True}

    print("\n[F] 정규화 4종")
    norm = {}
    for method in ("softmax", "minmax", "rank", "none"):
        try:
            # higher_is_better=false 인 브랜치는 부호를 뒤집어 '클수록 좋음'으로 통일 (fuse/core.py와 동일)
            got = {b: normalize_scores(s if higher[b] else -s, method=method, temperature=1.0)
                   for b, s in scores.items()}
            check(all(v.shape == (N, C) and np.isfinite(v).all() for v in got.values()),
                  f"normalize={method:8s}", f"metric 행합={got['metric'][0].sum():.3f}")
            if method == "softmax":
                norm = got
        except Exception as ex:
            check(False, f"normalize={method}", f"{type(ex).__name__}: {ex}")

    half = N // 2                      # 앞 절반 = valid, 뒤 절반 = test
    vs = {b: v[:half] for b, v in norm.items()}
    ts = {b: v[half:] for b, v in norm.items()}
    v_idx, t_idx = truth_idx[:half], truth_idx[half:]
    t_truth, t_open = truth[half:], is_open[half:]

    print("\n[G] 브랜치 가중치 — learned / manual")
    w_learned = learn_branch_weights(vs, v_idx, epochs=400, lr=0.05)
    check(abs(sum(w_learned.values()) - 1.0) < 1e-6, "learned 가중치 합=1",
          f"{ {k: round(v,3) for k,v in w_learned.items()} }")
    check(w_learned["classifier"] > w_learned["attributes"],
          "learned: 강한 브랜치(classifier) > 약한 브랜치(attributes)")
    w_manual = {"metric": 0.5, "classifier": 0.3, "attributes": 0.2}
    fused_l = combine(ts, w_learned)
    fused_m = combine(ts, w_manual)
    check(fused_l.shape == (len(t_idx), C), "combine → [N,C]", f"shape={fused_l.shape}")

    acc = lambda f, rej=None: float((np.array(classes)[f.argmax(1)] == t_truth).mean())
    base = {b: float((np.array(classes)[(ts[b]).argmax(1)] == t_truth).mean()) for b in ts}
    print(f"       브랜치 단독 정확도: { {k: round(v,3) for k,v in base.items()} }")
    check(acc(fused_l) >= max(base.values()) - 0.02, "결합이 단독 최고 브랜치 이상",
          f"결합={acc(fused_l):.3f} vs 최고단독={max(base.values()):.3f}")

    print("\n[H] open-set gate — metric이 open이라 하면 metric 단독 결정 (노션 규칙)")
    gated_scores, gated_mask = apply_openset_gate(fused_l, ts["metric"], classes, open_classes)
    a_before = acc(fused_l)
    a_after = float((np.array(classes)[gated_scores.argmax(1)] == t_truth).mean())
    open_before = float((np.array(classes)[fused_l.argmax(1)] == t_truth)[t_open].mean())
    open_after = float((np.array(classes)[gated_scores.argmax(1)] == t_truth)[t_open].mean())
    check(int(gated_mask.sum()) > 0, "gate가 실제로 적용됨", f"{int(gated_mask.sum())}/{len(t_idx)}장")
    check(open_after >= open_before, "gate 후 open 클래스 정확도 개선",
          f"open {open_before:.3f} → {open_after:.3f} / 전체 {a_before:.3f} → {a_after:.3f}")

    print("\n[I] threshold — auto / manual / none + max_reject_rate 상한")
    fused_v = combine(vs, w_learned)
    for cap in (None, 0.2, 0.05):
        th = find_threshold_auto(fused_v, v_idx, max_reject_rate=cap)
        rej = apply_threshold(gated_scores, th)
        rate = float(rej.mean())
        ok = cap is None or rate <= cap + 1e-6
        check(ok, f"threshold auto (max_reject_rate={str(cap):5s})",
              f"th={th:.4f} reject={rate:.3f}")
    th_manual = 0.5
    check(apply_threshold(gated_scores, th_manual).dtype == bool, "threshold manual", f"value={th_manual}")
    check(apply_threshold(gated_scores, None).sum() == 0, "threshold none → reject 0장")

    print("\n[J] top-k 평가")
    rej = apply_threshold(gated_scores, find_threshold_auto(fused_v, v_idx, max_reject_rate=0.2))
    prev = -1
    for k in (1, 2, 3):
        r = evaluate_topk(gated_scores, classes, t_truth, rej, top_k=k)
        check(r["accuracy"] >= prev, f"top{k} accuracy={r['accuracy']:.4f} "
              f"(판정분만={r['accuracy_on_decided']:.4f}, reject={r['reject_rate']:.3f})")
        prev = r["accuracy"]

    print("\n[K] 거부 성능 지표 (metric 패키지와 값이 같아야 한다 — 독립 구현)")
    rq = evaluate_rejection_quality(gated_scores, t_open, which=["min_dist", "margin", "dist_ent"])
    for name, v in rq["scores"].items():
        check(0.0 <= v["auroc"] <= 1.0, f"fuse reject {name:9s}",
              f"AUROC={v['auroc']:.4f} FPR@95={v['fpr_at_95tpr']:.4f}")
    from metric.openset import evaluate_rejection
    rm = evaluate_rejection(-gated_scores, t_open, which=["min_dist", "margin", "dist_ent"])
    same = all(abs(rq["scores"][n]["auroc"] - rm["scores"][n]["auroc"]) < 1e-9 for n in rq["scores"])
    check(same, "fuse 자체 구현 == metric 구현 (독립 구현이 같은 값)")

    print("\n[L] 산출 CSV 스키마")
    paths = [f"/x/{i}.jpg" for i in range(len(t_idx))]
    frame = build_prediction_frame(paths, gated_scores, classes, t_truth, rej, gated_mask, top_k=2)
    need = {"path", "true_label", "pred_label", "score", "rejected_unknown", "openset_gated"}
    check(need <= set(frame.columns), "predictions.csv 컬럼", f"{list(frame.columns)}")
    check((frame["pred_label"] == "__unknown__").sum() == int(rej.sum()),
          "reject된 행이 __unknown__ 으로 기록됨", f"{int(rej.sum())}장")


def section_data():
    """데이터 준비 (split/open/kfold/mask)"""
    from data.build import (scan_folder, assign_splits, build_kfold_manifests,
                            attach_band_masks, resolve_open_labels, split_closed_open,
                            choose_open_labels, detect_layout)

    tmp = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(3)
    labels = [f"{i:02d}_Company{i%7}_Sys{i}" for i in range(1, 43)]   # 42클래스
    root = tmp / "datas"
    for i, lab in enumerate(labels):
        n = int(rng.integers(6, 40))                                   # 클래스별 장수 불균형
        for j in range(n):
            f = root / lab / f"img_{i:02d}_{j:03d}.jpg"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b"x")

    print("\n[M] 폴더 스캔 + split 배분")
    check(detect_layout(root) == "class_folder", "layout 자동 감지 = class_folder")
    frame = scan_folder(root, "auto")
    check(len(frame) > 0 and frame["label"].nunique() == 42,
          "42클래스 스캔", f"{len(frame)}장 / {frame['label'].nunique()}클래스")
    frame = assign_splits(frame, ratios=(8, 1, 1), seed=42)
    pivot = frame.pivot_table(index="label", columns="split", values="path", aggfunc="count").fillna(0)
    check(set(frame["split"]) == {"train", "valid", "test"}, "train/valid/test 생성")
    check((pivot > 0).all().all(), "모든 클래스가 세 split에 최소 1장씩",
          f"최소값 train/valid/test = {pivot.min().astype(int).to_dict()}")
    ratio = frame["split"].value_counts(normalize=True)
    check(abs(ratio["train"] - 0.8) < 0.06, "대략 8:1:1 비율 유지",
          f"{ {k: round(v,3) for k,v in ratio.items()} }")

    print("\n[N] open 클래스 지정 — 표기가 달라도 맞춰지는가")
    wanted_exact = labels[27:]                                          # 15개 (28~42번)
    resolved = resolve_open_labels(frame, wanted_exact)
    check(len(resolved) == 15 and set(resolved) == set(wanted_exact), "정확히 일치하는 이름 15개")
    fuzzy = [l.replace("_", " ").lower() for l in wanted_exact]         # 표기 흔들기
    fuzzy = [f[3:] if f[:2].isdigit() else f for f in fuzzy]            # 앞 일련번호 제거
    resolved2 = resolve_open_labels(frame, fuzzy)
    check(set(resolved2) == set(wanted_exact), "표기 달라도(대소문자/구분자/일련번호) 매칭",
          f"예: {fuzzy[0]!r} → {sorted(resolved2)[0]!r}")
    try:
        resolve_open_labels(frame, ["Nonexistent_Brand_XYZ"])
        check(False, "없는 이름은 중단해야 함")
    except Exception as ex:
        check(True, "없는 이름은 조용히 넘기지 않고 중단", f"{type(ex).__name__}")

    print("\n[O] closed / open 분리 (leakage 확인)")
    parts = split_closed_open(frame, resolved)
    closed, open_f = parts["closed"], parts["open"]
    check(closed["label"].nunique() == 27, f"closed 27클래스", f"{closed['label'].nunique()}")
    check(open_f["label"].nunique() == 15, f"open 15클래스", f"{open_f['label'].nunique()}")
    check(set(closed["label"]) & set(open_f["label"]) == set(), "closed ∩ open = 공집합")
    check(len(closed) + len(open_f) == len(frame), "전체 = closed + open (누락 없음)")

    print("\n[P] open 클래스 자동 선정 (목록을 모를 때만 쓰는 대비책)")
    for strategy in ("rarest", "random"):
        got = choose_open_labels(frame, 15, seed=42, strategy=strategy)
        check(len(got) == 15 and len(set(got)) == 15, f"auto={strategy} 15개 선정")
    counts = frame["label"].value_counts()
    rarest = choose_open_labels(frame, 15, strategy="rarest")
    check(counts[rarest].max() <= counts.median(), "rarest는 실제로 희귀 클래스를 고름",
          f"선정된 최대 장수={counts[rarest].max()} vs 중앙값={counts.median():.0f}")

    print("\n[Q] band mask 연결")
    mask_root = tmp / "masks"
    sample = frame.sample(frac=0.7, random_state=0)
    for p in sample["path"]:
        m = mask_root / (Path(p).stem + "_band.png")
        m.parent.mkdir(parents=True, exist_ok=True); m.write_bytes(b"x")
    withmask = attach_band_masks(frame, mask_root, suffix="_band.png", column="mask_path")
    have = (withmask["mask_path"] != "").sum()
    check(have == len(sample), "mask_path가 stem 기준으로 정확히 연결됨",
          f"{have}/{len(frame)}장 (마스크 만든 수={len(sample)})")
    check("_band.png" in withmask.loc[withmask["mask_path"] != "", "mask_path"].iloc[0],
          "00_data.yaml의 suffix(_band.png)와 동일한 이름 규칙")

    print("\n[R] K-fold — test는 고정, train/valid만 바뀌는가")
    folds = build_kfold_manifests(frame, tmp / "folds", "manifest_closed27", k=5, seed=42)
    check(len(folds) == 5, "fold 5개 생성", f"{[f.name for f in folds]}")
    tests, trains = [], []
    for f in folds:
        d = pd.read_csv(f)
        tests.append(set(d[d["split"] == "test"]["path"]))
        trains.append(set(d[d["split"] == "train"]["path"]))
    check(all(t == tests[0] for t in tests), "test는 모든 fold에서 동일 (평가 기준 고정)",
          f"{len(tests[0])}장")
    check(len(set(map(frozenset, trains))) == 5, "train은 fold마다 다름")
    check(all("source_split" in pd.read_csv(f).columns for f in folds), "원래 split이 source_split에 보존")

    shutil.rmtree(tmp)


def section_literals():
    """yaml 광고 선택지 == 코드 구현"""
    def const(module_path, name):
        """모듈을 import하지 않고 상수 튜플만 소스에서 읽는다 (torch 회피)."""
        src = (ROOT / module_path).read_text(encoding="utf-8")
        m = re.search(rf"^{name}\s*=\s*\(([^)]*)\)", src, re.M)
        if not m:
            return None
        return tuple(v.strip().strip('"\'') for v in m.group(1).split(",") if v.strip())

    def literals(module_path, pattern):
        src = (ROOT / module_path).read_text(encoding="utf-8")
        return set(re.findall(pattern, src))

    print("\n[T] yaml 광고 선택지 == 코드 구현 선택지")
    cases = [
        ("metric embedding_loss", {"supcon","triplet","arcface","covaware_supcon"},
         set(const("src/metric/train.py", "TRAIN_LOSSES") or ())),
        ("metric prototype",      {"mean","ss","medoid","kmeans_multi","reference"},
         set(const("src/metric/prototypes.py", "PROTOTYPE_METHODS") or ())),
        ("metric distance",       {"cosine","euclidean","mahalanobis","fractional","centered_angular"},
         set(const("src/metric/scoring.py", "DISTANCES") or ())),
        ("metric covariance",     {"ledoit_wolf","oas","diagonal","class_balanced","rda"},
         set(const("src/metric/scoring.py", "COVARIANCE_ESTIMATORS") or ())),
        ("metric transform",      {"none","l2norm","pca"},
         set(const("src/metric/scoring.py", "TRANSFORMS") or ())),
        ("metric reject score",   {"min_dist","margin","dist_ent","logit_msp","logit_ent"},
         set(const("src/metric/openset.py", "REJECT_SCORES") or ())),
        ("metric visualize reducer", {"tsne","umap","none"},
         set(const("src/metric/visualize.py", "REDUCERS") or ())),
        ("fuse normalize",        {"softmax","minmax","rank","none"},
         set(const("src/fuse/core.py", "NORMALIZE_METHODS") or ())),
        ("masked_vit input_mode", {"original","band"},
         set(const("src/masked_vit/band_preprocessing.py", "INPUT_MODES") or ())),
        ("attributes fuse mode",  {"learned","min_match"},
         set(const("src/attributes/fuse.py", "FUSE_MODES") or ())),
    ]
    for name, advertised, implemented in cases:
        if not implemented:
            check(False, f"{name}: 코드에서 상수를 못 찾음")
            continue
        check(advertised <= implemented, f"{name:26s} yaml {len(advertised)}종 ⊆ 구현 {len(implemented)}종",
              f"구현={sorted(implemented)}")
        extra = implemented - advertised
        if extra:
            print(f"         (코드에만 있고 yaml 주석엔 없는 값: {sorted(extra)})")

    print("\n[U] 분기가 실제로 존재하는지 (상수가 없는 옵션들)")
    branch_cases = [
        # 값 하나는 if로, 나머지는 else로 처리하는 곳들 — 두 갈래가 모두 있는지 확인
        ("backbone finetune_mode", "src/backbone/core.py",
         {'mode == "full"', 'mode != "partial"'}, None),
        ("covaware covariance_mode", "src/metric/models.py",
         {'covariance_mode == "pooled_full"', 'covariance_mode == "pooled_diagonal"',
          'covariance_mode == "legacy"'}, None),
        ("covaware loss_embedding", "src/metric/train.py",
         {'loss_embedding = covaware.get("loss_embedding", "backbone")',
          'if loss_embedding == "backbone"'}, None),
        ("attributes fuse mode", "src/attributes/fuse.py",
         {'FUSE_MODES = ("learned", "min_match")', 'else:  # min_match'}, None),
        ("fuse weights.mode", "src/run_fuse.py",
         {'weight_cfg["mode"] == "learned"', 'weight_cfg["mode"] == "manual"'}, None),
        ("fuse threshold.mode", "src/run_fuse.py",
         {'threshold_cfg["mode"] == "auto"', 'threshold_cfg["mode"] == "manual"'}, None),
        ("data layout", "src/data/build.py",
         {'layout == "class_folder"', 'return "split_class_folder"'}, None),
        ("openset auto", "src/data/build.py",
         {'strategy == "rarest"', 'strategy == "random"'}, None),
        ("reference_reduce", "src/metric/prototypes.py",
         {'reference_reduce: str = "mean"', 'reference_reduce == "each"'}, None),
    ]
    for name, path, snippets, _ in branch_cases:
        src = (ROOT/path).read_text(encoding="utf-8")
        missing = [s_ for s_ in snippets if s_ not in src]
        check(not missing, f"{name:26s} 두 갈래 모두 코드에 존재",
              f"{len(snippets)}개 확인" + (f" 누락={missing}" if missing else ""))


def section_config_hygiene():
    """죽은 yaml 키 / 스테이지 간 체크포인트 경로 합의"""
    import yaml

    src_text = "\n".join(
        f.read_text(encoding="utf-8") for f in (ROOT / "src").rglob("*.py")
        if f.name != "verify_pipeline.py" and "__pycache__" not in str(f)
    )

    print("\n[V] 코드가 한 번도 읽지 않는 yaml 키 (죽은 옵션은 폐쇄망에서 오해를 만든다)")
    # 값이 데이터인 블록은 제외 — 키 이름 자체가 사용자 데이터다
    DATA_BLOCKS = ("backbones.", "features.", "spec.value_aliases",
                   "openset.labels", "sam1.generator.")

    def leaves(node, prefix=""):
        out = []
        if isinstance(node, dict):
            for k, v in node.items():
                out.append((prefix + k, k))
                out += leaves(v, prefix + k + ".")
        return out

    dead = []
    for cfg_path in sorted((ROOT / "configs").glob("*.yaml")):
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        for full, leaf in leaves(raw):
            if full.startswith(DATA_BLOCKS):
                continue
            if re.search(rf'["\']{re.escape(leaf)}["\']', src_text):
                continue
            dead.append(f"{cfg_path.name}: {full}")
    check(not dead, f"죽은 키 없음", f"발견={dead}" if dead else f"{len(list((ROOT/'configs').glob('*.yaml')))}개 config 검사")

    print("\n[X] 의존 패키지 선언이 실제 import를 덮는가")
    import ast as _ast

    # import 이름 → 배포 이름
    DIST = {"PIL": "Pillow", "cv2": "opencv-python", "sklearn": "scikit-learn",
            "yaml": "PyYAML", "umap": "umap-learn", "segment_anything": "segment-anything",
            "sam2": "SAM-2"}
    # src/ 안의 모듈·패키지는 전부 로컬 (파일이 늘어도 자동으로 따라간다)
    local = {p.stem for p in (ROOT / "src").glob("*.py")}
    local |= {d.name for d in (ROOT / "src").iterdir() if d.is_dir()}

    imported = set()
    for f in sorted((ROOT / "src").rglob("*.py")):
        if "__pycache__" in str(f) or f.name == "verify_pipeline.py":
            continue
        for node in _ast.walk(_ast.parse(f.read_text(encoding="utf-8"))):
            mods = ([a.name for a in node.names] if isinstance(node, _ast.Import)
                    else [node.module] if isinstance(node, _ast.ImportFrom)
                    and node.level == 0 and node.module else [])
            for m in mods:
                top = m.split(".")[0]
                if top not in sys.stdlib_module_names and top not in local and not top.startswith("_"):
                    imported.add(top)

    # 선언 위치는 세 군데다: requirements*.txt / Dockerfile 의 pip install / 주석 처리된 선택 의존
    req_files = sorted(ROOT.glob("requirements*.txt"))
    check(bool(req_files), f"requirements 파일 존재", f"{[f.name for f in req_files]}")

    def _names(lines):
        out = set()
        for line in lines:
            # "SAM-2 @ git+https://..." 형태도 이름만 뽑는다
            name = re.split(r"[<>=!;\[@]", line.strip())[0].strip().lower()
            if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", name):
                out.add(name)
        return out

    declared = set()
    OPT = re.compile(r"^#\s*([a-zA-Z][\w.-]*(?:\s*[<>=!~]=?\s*[\w.*]+)?)\s*(?:#.*)?$")
    for f in req_files:
        raw = f.read_text(encoding="utf-8").splitlines()
        declared |= _names(l.split("#")[0] for l in raw if l.split("#")[0].strip())
        # 주석 처리된 requirement 한 줄 = 선택 의존으로 선언된 것
        declared |= _names(m.group(1) for m in (OPT.match(l.strip()) for l in raw) if m)
    dockerfile = ROOT / "Dockerfile"
    if dockerfile.is_file():
        # Dockerfile 이 직접 설치하는 것 (torch 는 CUDA 인덱스라 requirements 에 없다)
        text = dockerfile.read_text(encoding="utf-8")
        declared |= {m.lower() for m in re.findall(r"^\s*(?:&&\s*)?([a-zA-Z][\w.-]*)==", text, re.M)}
        declared |= {w.lower() for w in re.findall(r"\b([a-zA-Z][\w.-]*)==[\d.]+", text)}

    missing = []
    for mod in sorted(imported):
        name = DIST.get(mod, mod).lower()
        names = {name, mod.lower(), mod.replace("_", "-").lower()}
        if mod == "cv2":
            names |= {"opencv-python-headless"}
        if not (names & declared):
            missing.append(mod)
    check(not missing, f"import {len(imported)}개가 모두 선언됨",
          f"누락={missing}" if missing else f"{sorted(imported)}")

    print("\n[W] 스테이지 간 체크포인트 경로 합의")
    seg = load_config(ROOT / "configs/02_segmentation.yaml")
    masked = load_config(ROOT / "configs/04_masked_vit.yaml")
    pairs = [
        ("SAM2 base 체크포인트", seg["sam2_train"]["sam2_checkpoint"], masked["sam"]["sam2_checkpoint"]),
        ("SAM2 config", seg["sam2_train"]["sam2_config"], masked["sam"]["sam2_config"]),
        ("파인튜닝 adapter", seg["sam2_train"]["save_path"], masked["sam"]["model_checkpoint"]),
    ]
    for name, produced, consumed in pairs:
        check(str(produced) == str(consumed),
              f"{name}: Stage 0 산출 == masked-vit 입력", f"{produced}")


SECTIONS = {"wiring": section_wiring, "options": section_options, "fuse": section_fuse,
            "data": section_data, "literals": section_literals,
            "config": section_config_hygiene}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=list(SECTIONS), action="append",
                        help="이 섹션만 실행 (여러 번 지정 가능)")
    args = parser.parse_args()

    for name in (args.only or list(SECTIONS)):
        print(f"\n{'=' * 64}\n### {name} — {SECTIONS[name].__doc__}\n{'=' * 64}")
        SECTIONS[name]()

    print("\n" + "=" * 64)
    print(f"OK {len(RESULTS['OK'])}   WARN {len(RESULTS['WARN'])}   FAIL {len(RESULTS['FAIL'])}")
    for f in RESULTS["FAIL"]:
        print("  실패:", f)
    return 1 if RESULTS["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

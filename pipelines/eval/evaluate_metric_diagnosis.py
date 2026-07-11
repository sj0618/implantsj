#!/usr/bin/env python3
"""Compact 42-label metric-representation diagnosis."""
import argparse, hashlib, json, re, sys, unicodedata, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import IMAGE_EXTENSIONS, ImageClassificationDataset, build_transforms, load_manifest
from src.metrics import compute_metrics
from src.models import MetricModel, create_classifier, load_matching_weights, unwrap_state_dict

MODELS = ["ViT Base", "ViT-ClosedFT", "ViT-ClosedFT-SupCon", "ViT-FullFT", "ViT-FullFT-SupCon"]
PROTOS, DISTS, SPLITS = ["gt", "mean", "ss"], ["euclidean", "cosine"], ["valid", "test"]

def normalize_label_name(value):
    value = unicodedata.normalize("NFKC", str(value)).strip()
    value = re.sub(r"^\s*\d+\s*(?:[._:\-)]\s*)+", "", value)
    spaced = re.match(r"^\s*(\d+)\s+(.+)$", value)
    if spaced and int(spaced.group(1)) <= 42: value = spaced.group(2)
    value = re.sub(r"\s*\([^()]*\)\s*$", "", value)
    return "".join(c for c in value.casefold() if c.isalnum())

def map_gt_label_folders(root, labels, report_csv=None):
    root = Path(root); found = {}
    for folder in sorted(p for p in root.rglob("*") if p.is_dir()):
        found.setdefault(normalize_label_name(folder.name), []).append(folder)
    mapped, rows, seen = {}, [], {}
    for label in sorted(map(str, labels)):
        key = normalize_label_name(label)
        if key in seen: raise RuntimeError(f"Label collision: {seen[key]}, {label}")
        seen[key] = label; matches = found.get(key, [])
        images = sorted({p.resolve() for f in matches for p in f.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS})
        status = "matched" if images else "unmatched"
        rows.append({"label":label, "normalized_label":key, "status":status, "folder":" | ".join(map(str,matches)), "image_count":len(images)})
        if status == "matched": mapped[label] = images
    if report_csv: pd.DataFrame(rows).to_csv(report_csv, index=False)
    bad = [r for r in rows if r["status"] != "matched"]
    if bad: raise RuntimeError(f"GT mapping failed; see {report_csv}: {bad[:5]}")
    return mapped, rows

def choose_support_indices(labels, paths, policy, count, seed):
    if policy != "ss": return np.arange(len(labels))
    rng, out = np.random.default_rng(seed), []
    for label in sorted(set(labels.tolist())):
        idx = np.flatnonzero(labels == label); idx = idx[np.argsort(paths[idx].astype(str))]
        out += rng.choice(idx, min(count, len(idx)), replace=False).tolist()
    return np.asarray(sorted(out), dtype=np.int64)

def build_prototypes(features, labels, indices):
    selected = labels[indices]; names = sorted(set(selected.tolist()))
    return np.stack([features[indices[selected == name]].mean(0) for name in names]), names

def prototype_predict(query, prototypes, names, distance):
    if distance == "cosine":
        q = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
        p = prototypes / np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)
        scores = q @ p.T; best = scores.argmax(1)
    else:
        squared = np.maximum((query*query).sum(1,keepdims=True) + (prototypes*prototypes).sum(1)[None,:] - 2*query@prototypes.T, 0)
        best = squared.argmin(1); scores = np.sqrt(squared)
    return np.asarray([names[i] for i in best], dtype=object), scores[np.arange(len(best)), best]

def digest(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1048576),b""): h.update(block)
    return h.hexdigest()

def clean(state):
    out = {}
    for key,value in state.items():
        for prefix in ["module.","model."]:
            if key.startswith(prefix): key = key[len(prefix):]
        out[key] = value
    return out

def record(path):
    try:
        ck = torch.load(path,map_location="cpu"); args = ck.get("args",{}); classes = ck.get("class_to_idx",{})
        state = clean(unwrap_state_dict(ck)); manifest = Path(args.get("manifest") or "")
        manifest_labels, manifest_splits, manifest_error = [], {}, ""
        if manifest.is_file():
            try:
                frame = load_manifest(manifest); manifest_labels = sorted(frame.label.astype(str).unique()); manifest_splits = frame.split.astype(str).value_counts().to_dict()
            except Exception as e: manifest_error = repr(e)
        return {"checkpoint":str(path.resolve()), "kind":"supcon" if any(k.startswith("projector.") for k in state) else "classifier",
                "model_name":args.get("model_name",""), "loss_name":args.get("loss_name",""), "labels":sorted(map(str,classes)),
                "manifest":str(manifest), "manifest_hash":digest(manifest) if manifest.is_file() else "", "manifest_labels":manifest_labels, "manifest_splits":manifest_splits, "manifest_error":manifest_error, "epoch":ck.get("epoch",0),
                "score":(ck.get("metrics",{}) or {}).get("macro_f1",-1), "readable":True}
    except Exception as e: return {"checkpoint":str(path), "readable":False, "error":repr(e)}

def candidate_facts(item, target):
    checkpoint, manifest = set(item.get("labels",[])), set(item.get("manifest_labels",[]))
    if manifest:
        return checkpoint == manifest == target, True, "own checkpoint manifest exact"
    return checkpoint == target, False, "checkpoint labels exact"

def classifier_features(model, x):
    z = model.forward_features(x)
    if hasattr(model,"forward_head"):
        try: z = model.forward_head(z,pre_logits=True)
        except TypeError: z = model.forward_head(z)
    elif z.ndim == 3: z = z[:,0]
    return z if z.ndim == 2 else torch.nn.functional.adaptive_avg_pool2d(z,1).flatten(1)

def make_model(item, device):
    kind = item["kind"]
    if kind == "pretrained_base":
        name, size = item["model_name"], int(item.get("image_size",224))
        model = create_classifier(name,0,True); forward = lambda x:classifier_features(model,x)
        return model.to(device).eval(), forward, size
    path=Path(item["checkpoint"]); ck = torch.load(path,map_location="cpu"); args = ck.get("args",{}); state = clean(unwrap_state_dict(ck))
    name, size = args.get("model_name",item.get("model_name","vit_base_patch16_224")), int(args.get("image_size",224))
    if kind == "supcon":
        dim = int(args.get("embedding_dim") or state["projector.3.weight"].shape[0])
        model = MetricModel(name,dim,False,float(args.get("dropout",0)),float(args.get("drop_path_rate",0)))
        model.load_state_dict(state); forward = lambda x:model.projector(model.forward_features(x))
    elif kind == "classifier":
        model = create_classifier(name,len(ck.get("class_to_idx",{})),False,float(args.get("dropout",0)),float(args.get("drop_path_rate",0)))
        model.load_state_dict(state); forward = lambda x:classifier_features(model,x)
    else:
        model = create_classifier(name,0,False); loaded,_ = load_matching_weights(model,path)
        if loaded < 10: raise RuntimeError(f"Invalid Base checkpoint: {path}")
        forward = lambda x:classifier_features(model,x)
    return model.to(device).eval(), forward, size

@torch.no_grad()
def extract(model, forward, manifest, size, batch, workers, device):
    df = load_manifest(manifest); labels = sorted(df.label.astype(str).unique()); mapping = {v:i for i,v in enumerate(labels)}
    ds = ImageClassificationDataset(manifest,None,mapping,build_transforms(size,False),True)
    loader = DataLoader(ds,batch_size=batch,shuffle=False,num_workers=workers,pin_memory=device.type=="cuda")
    z,paths = [],[]
    for x,_,p in loader: z.append(forward(x.to(device)).cpu().numpy()); paths += list(p)
    return {"features":np.concatenate(z), "labels":df.label.astype(str).to_numpy(object),
            "splits":df.split.astype(str).to_numpy(object), "paths":np.asarray(paths,dtype=object)}

def stratified_rows(rows, limit, seed):
    if limit <= 0 or len(rows) <= limit: return rows
    groups = {}
    for i,row in enumerate(rows): groups.setdefault(row[:3],[]).append(i)
    rng=np.random.default_rng(seed); queues=[]
    for key in sorted(groups):
        idx=np.asarray(groups[key]); rng.shuffle(idx); queues.append(idx.tolist())
    chosen=[]
    while len(chosen)<limit:
        moved=False
        for queue in queues:
            if queue and len(chosen)<limit: chosen.append(queue.pop()); moved=True
        if not moved: break
    return [rows[i] for i in chosen]

def save_tsne(name, caches, out, seed, limit):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    rows = [(part,c["labels"][i],c["splits"][i],c["paths"][i],c["features"][i]) for part,c in caches.items() for i in range(len(c["labels"]))]
    total=len(rows); rows=stratified_rows(rows,limit,seed)
    x = np.stack([r[4] for r in rows]); x = PCA(min(50,x.shape[1],len(x)-1),random_state=seed).fit_transform(x) if x.shape[1]>50 else x
    xy = TSNE(2,perplexity=min(30,max(1,(len(x)-1)/3)),init="pca",learning_rate="auto",random_state=seed).fit_transform(x)
    df = pd.DataFrame({"partition":[r[0] for r in rows],"label":[r[1] for r in rows],"split":[r[2] for r in rows],"path":[r[3] for r in rows],"x":xy[:,0],"y":xy[:,1]})
    stem = name.replace(" ","_"); df.to_csv(out/f"{stem}.csv",index=False); np.savez_compressed(out/f"{stem}.npz",features=x,coords=xy,**{c:df[c].to_numpy() for c in ["partition","label","split","path"]})
    counts={"/".join(map(str,k)):int(v) for k,v in df.groupby(["partition","split","label"]).size().items()}; meta={"total_features":total,"sampled":len(df),"max_samples":limit,"seed":seed,"method":"deterministic round-robin by partition/split/label","counts":counts}; (out/f"{stem}.sampling.json").write_text(json.dumps(meta,indent=2))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    label_names=sorted(df.label.unique()); colors={label:plt.cm.turbo(i/max(1,len(label_names)-1)) for i,label in enumerate(label_names)}
    def panels(splits, path):
        fig,axes=plt.subplots(2,len(splits),figsize=(7*len(splits),11),squeeze=False)
        for row,part in enumerate(["closed","open"]):
            for col,split in enumerate(splits):
                ax=axes[row,col]; q=df[(df.partition==part)&(df.split==split)]
                for label,g in q.groupby("label",sort=True): ax.scatter(g.x,g.y,s=9,alpha=.65,color=colors[label])
                ax.set_title(f"{name}: {part} / {split}"); ax.set_xticks([]); ax.set_yticks([])
        handles=[Line2D([0],[0],marker="o",linestyle="",markersize=5,color=colors[label],label=label) for label in label_names]
        fig.legend(handles=handles,loc="center left",bbox_to_anchor=(.99,.5),fontsize=7); fig.tight_layout(rect=(0,0,.86,1)); fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)
    panels(["train","test"],out/f"{stem}.png"); panels(["valid"],out/f"{stem}_valid.png")
    return meta

def save_curve(name, item, out):
    if not item.get("checkpoint"): return {"model":name,"history":"","status":"not_applicable","png":""}
    history=Path(item["checkpoint"]).parent.parent/"metrics"/"history.csv"; stem=name.replace(" ","_")
    if not history.is_file(): return {"model":name,"history":str(history),"status":"missing","png":""}
    df=pd.read_csv(history); copy=out/f"{stem}.csv"; df.to_csv(copy,index=False)
    groups=[("loss",["train_loss"]),("accuracy",[c for c in ["valid_accuracy","test_accuracy"] if c in df]),("macro-F1",[c for c in ["valid_macro_f1","test_macro_f1"] if c in df])]; groups=[g for g in groups if g[1] and all(c in df for c in g[1])]
    png=""
    if groups:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        x=df["epoch"] if "epoch" in df else np.arange(1,len(df)+1); fig,axes=plt.subplots(1,len(groups),figsize=(5*len(groups),4),squeeze=False)
        for ax,(title,columns) in zip(axes[0],groups):
            for column in columns: ax.plot(x,df[column],marker="o",label=column)
            ax.set_title(title); ax.set_xlabel("epoch"); ax.legend()
        fig.tight_layout(); target=out/f"{stem}.png"; fig.savefig(target,dpi=180); plt.close(fig); png=str(target)
    missing=sorted({"test_accuracy","test_macro_f1"}-set(df.columns)); status="partial" if missing else "ok"
    reason=f"history lacks per-epoch {', '.join(missing)}; available curves only" if missing else ""
    return {"model":name,"history":str(history),"status":status,"csv":str(copy),"png":png,"reason":reason}

def save_partial(out, summary, predictions, members, curves, progress):
    (pd.DataFrame(summary) if summary else pd.DataFrame(columns=["model","prototype","distance","partition","split","status","reason"])).to_csv(out/"summary.csv",index=False)
    (pd.DataFrame(predictions) if predictions else pd.DataFrame(columns=["model","prototype","distance","partition","split","path","true_label","pred_label","score","correct"])).to_csv(out/"predictions.csv",index=False)
    (pd.DataFrame(members) if members else pd.DataFrame(columns=["model","partition","prototype","label","path"])).to_csv(out/"prototype_membership.csv",index=False)
    (pd.DataFrame(curves) if curves else pd.DataFrame(columns=["model","history","status","csv","png","reason"])).to_csv(out/"curve_inventory.csv",index=False)
    (out/"progress.json").write_text(json.dumps(progress,indent=2))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--closed-manifest",required=True); p.add_argument("--open-manifest",required=True); p.add_argument("--full-manifest",required=True)
    p.add_argument("--checkpoint-root",action="append",required=True); p.add_argument("--base-checkpoint"); p.add_argument("--base-model-name",default="vit_large_patch16_224.augreg_in21k_ft_in1k")
    p.add_argument("--closed-checkpoint"); p.add_argument("--closed-supcon-checkpoint"); p.add_argument("--full-checkpoint"); p.add_argument("--full-supcon-checkpoint")
    p.add_argument("--gt-zip"); p.add_argument("--gt-root"); p.add_argument("--output-dir",required=True)
    p.add_argument("--ss-per-label",type=int,default=5); p.add_argument("--seed",type=int,default=42); p.add_argument("--batch-size",type=int,default=64); p.add_argument("--num-workers",type=int,default=8); p.add_argument("--tsne-max-samples",type=int,default=0); p.add_argument("--allow-cpu",action="store_true"); p.add_argument("--inventory-only",action="store_true"); a=p.parse_args()
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); (out/"features").mkdir(exist_ok=True); (out/"tsne").mkdir(exist_ok=True); (out/"curves").mkdir(exist_ok=True)
    manifests={"closed":Path(a.closed_manifest),"open":Path(a.open_manifest),"full":Path(a.full_manifest)}; frames={k:load_manifest(v) for k,v in manifests.items()}; labels={k:set(v.label.astype(str)) for k,v in frames.items()}
    errors=[]; split_errors={}
    if len(labels["closed"])!=27: errors.append("closed label count is not 27")
    if len(labels["open"])!=15: errors.append("open label count is not 15")
    if len(labels["full"])!=42: errors.append("full label count is not 42")
    if labels["closed"]&labels["open"]: errors.append("closed/open overlap")
    if labels["closed"]|labels["open"]!=labels["full"]: errors.append("closed/open union differs from full")
    for part,frame in frames.items():
        if "split" not in frame: errors.append(f"{part} manifest has no split column"); continue
        missing={label:sorted(set(["train","valid","test"])-set(group.split.astype(str))) for label,group in frame.groupby("label")}; missing={k:v for k,v in missing.items() if v}
        if missing: split_errors[part]=missing; errors.append(f"{part} labels missing train/valid/test splits")
    audit={"ok":not errors,"errors":errors,"split_errors":split_errors,"manifests":{k:{"path":str(v.resolve()),"hash":digest(v),"rows":len(frames[k]),"labels":len(labels[k]),"splits":frames[k].split.value_counts().to_dict()} for k,v in manifests.items()}}; (out/"manifest_audit.json").write_text(json.dumps(audit,indent=2))
    if errors: raise RuntimeError(errors)
    paths=set()
    for root in map(Path,a.checkpoint_root): paths.update(root.rglob("best.pt")); paths.update(root.rglob("best*.pth"))
    explicit={"ViT-ClosedFT":a.closed_checkpoint,"ViT-ClosedFT-SupCon":a.closed_supcon_checkpoint,"ViT-FullFT":a.full_checkpoint,"ViT-FullFT-SupCon":a.full_supcon_checkpoint}
    paths.update(Path(path) for path in explicit.values() if path)
    records=[record(x) for x in sorted(paths)]; pd.DataFrame(records).to_csv(out/"checkpoint_inventory.csv",index=False); selected={}; selection_errors={}
    needs={"ViT-ClosedFT":("classifier","closed"),"ViT-ClosedFT-SupCon":("supcon","closed"),"ViT-FullFT":("classifier","full"),"ViT-FullFT-SupCon":("supcon","full")}
    for name,(kind,part) in needs.items():
        candidates=[]
        for r in records:
            eligible,own,basis=candidate_facts(r,labels[part])
            if r.get("readable") and r.get("kind")==kind and r.get("model_name","").startswith("vit_") and (kind!="supcon" or r.get("loss_name") in ["","supcon"]) and eligible: candidates.append((r,own,basis))
        if explicit[name]:
            wanted=str(Path(explicit[name]).resolve()); matches=[x for x in candidates if str(Path(x[0]["checkpoint"]).resolve())==wanted]
            if matches: candidates=matches
            else: selection_errors[name]=f"explicit checkpoint failed exact {part} validation: {wanted}"; candidates=[]
        if candidates:
            chosen,own,basis=max(candidates,key=lambda x:(x[1],set(["train","valid","test"])<=set(x[0]["manifest_splits"]),x[0]["epoch"],x[0]["score"]))
            selected[name]={**chosen,"selection_basis":("explicit " if explicit[name] else "")+basis,"missing_target_labels":[],"own_manifest_backed":own,"own_manifest_complete_splits":set(["train","valid","test"])<=set(chosen["manifest_splits"])}
    selected["ViT Base"]={"kind":"base","checkpoint":str(Path(a.base_checkpoint).resolve()),"model_name":a.base_model_name,"selection_basis":"explicit base checkpoint"} if a.base_checkpoint else {"kind":"pretrained_base","model_name":a.base_model_name,"image_size":224,"selection_basis":"timm pretrained weights"}
    (out/"checkpoint_selection.json").write_text(json.dumps(selected,indent=2)); (out/"checkpoint_selection_errors.json").write_text(json.dumps(selection_errors,indent=2)); print("SELECTED=",",".join(x for x in MODELS if x in selected)); print("MISSING=",",".join(x for x in MODELS if x not in selected))
    if a.inventory_only: return
    if not a.gt_zip or not a.gt_root: raise RuntimeError("GT paths required")
    if a.ss_per_label < 1: raise RuntimeError("ss-per-label must be positive")
    gtroot=Path(a.gt_root); gtroot.mkdir(parents=True,exist_ok=True)
    if not any(gtroot.iterdir()):
        with zipfile.ZipFile(a.gt_zip) as z:
            for n in z.namelist():
                if (gtroot/n).resolve()!=gtroot.resolve() and gtroot.resolve() not in (gtroot/n).resolve().parents: raise RuntimeError(f"Unsafe zip: {n}")
            z.extractall(gtroot)
    gt,_=map_gt_label_folders(gtroot,labels["full"],out/"gt_mapping.csv"); gtmanifest=out/"gt_manifest.csv"; pd.DataFrame([{"path":str(p),"label":l,"split":"gt"} for l,ps in gt.items() for p in ps]).to_csv(gtmanifest,index=False)
    if set(gt)!=labels["full"]: raise RuntimeError("GT prototype coverage differs from full labels")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type=="cpu" and not a.allow_cpu: raise RuntimeError("CUDA unavailable")
    summary,predictions,members,curves=[],[],[],[]; progress={"expected_result_rows":120,"tsne_max_samples":a.tsne_max_samples,"models":{}}
    for name in MODELS:
        local_summary=[]; local_predictions=[]; local_members=[]; reason=""
        try:
            if name not in selected: raise RuntimeError(selection_errors.get(name,"checkpoint missing"))
            item=selected[name]; curves.append(save_curve(name,item,out/"curves")); model,forward,size=make_model(item,device); caches={}; folder=out/"features"/name.replace(" ","_"); folder.mkdir(exist_ok=True)
            for part in ["closed","open"]:
                caches[part]=extract(model,forward,manifests[part],size,a.batch_size,a.num_workers,device); np.savez_compressed(folder/f"{part}.npz",**caches[part])
            gtc=extract(model,forward,gtmanifest,size,a.batch_size,a.num_workers,device); np.savez_compressed(folder/"gt.npz",**gtc); tsne_meta=save_tsne(name,caches,out/"tsne",a.seed,a.tsne_max_samples)
            for part,c in caches.items():
                classes=sorted(set(c["labels"].tolist())); label_to_idx={v:i for i,v in enumerate(classes)}
                for policy in PROTOS:
                    support=gtc if policy=="gt" else c; required_split="gt" if policy=="gt" else "train"; pool=np.flatnonzero(np.isin(support["labels"],classes)&(support["splits"]==required_split))
                    covered=set(support["labels"][pool].tolist()); missing=sorted(set(classes)-covered)
                    if missing: raise RuntimeError(f"{part}/{policy} prototype coverage missing {missing}")
                    chosen=pool[choose_support_indices(support["labels"][pool],support["paths"][pool],policy,a.ss_per_label,a.seed)]; prototypes,prototype_names=build_prototypes(support["features"],support["labels"],chosen)
                    if set(prototype_names)!=set(classes): raise RuntimeError(f"{part}/{policy} prototype labels differ from query labels")
                    local_members += [{"model":name,"partition":part,"prototype":policy,"label":support["labels"][i],"path":support["paths"][i]} for i in chosen]
                    for distance in DISTS:
                        for split in SPLITS:
                            qi=np.flatnonzero(c["splits"]==split)
                            if not len(qi): raise RuntimeError(f"{part}/{split} has no queries")
                            guess,score=prototype_predict(c["features"][qi],prototypes,prototype_names,distance); true=c["labels"][qi]; metrics=compute_metrics([label_to_idx[x] for x in true],[label_to_idx[x] for x in guess],classes)
                            local_summary.append({"model":name,"prototype":policy,"distance":distance,"partition":part,"split":split,"status":"ok","reason":"",**{k:metrics[k] for k in ["accuracy","macro_f1","weighted_f1"]}}); local_predictions += [{"model":name,"prototype":policy,"distance":distance,"partition":part,"split":split,"path":c["paths"][j],"true_label":true[i],"pred_label":guess[i],"score":float(score[i]),"correct":bool(true[i]==guess[i])} for i,j in enumerate(qi)]
            if len(local_summary)!=24: raise RuntimeError(f"expected 24 result rows, got {len(local_summary)}")
            summary+=local_summary; predictions+=local_predictions; members+=local_members; progress["models"][name]={"status":"ok","result_rows":24,"tsne":tsne_meta}
        except Exception as e:
            reason=repr(e); summary += [{"model":name,"prototype":p,"distance":d,"partition":q,"split":s,"status":"unavailable","reason":reason} for p in PROTOS for d in DISTS for q in ["closed","open"] for s in SPLITS]; progress["models"][name]={"status":"unavailable","reason":reason,"result_rows":24}
            if not any(r["model"]==name for r in curves): curves.append({"model":name,"history":"","status":"unavailable","png":"","reason":reason})
        finally:
            if "forward" in locals(): del forward
            if "model" in locals(): del model
            if "support" in locals(): del support
            if "c" in locals(): del c
            if "caches" in locals(): del caches
            if "gtc" in locals(): del gtc
            if device.type=="cuda": torch.cuda.empty_cache()
            save_partial(out,summary,predictions,members,curves,progress)
    result=pd.DataFrame(summary); ok=result[result.status=="ok"]
    tsne_note="all features" if a.tsne_max_samples<=0 else f"deterministic stratified sample, maximum {a.tsne_max_samples} per model"
    lines=["# Metric Learning Diagnosis","",f"Artifacts: `{out.resolve()}`",f"Seed: {a.seed}; SS: {a.ss_per_label}; GT unmatched: 0",f"t-SNE: {tsne_note}; full features remain in NPZ",f"SupCon: raw projector outputs are retained so Euclidean and cosine remain distinct","","## Mean performance by partition"]
    for partition in ["closed","open"]:
        lines += ["",f"### {partition.title()}"]
        for name in MODELS:
            q=ok[(ok.model==name)&(ok.partition==partition)]; lines.append(f"- {name}: "+("unavailable" if q.empty else f"accuracy {q.accuracy.mean():.4f}, macro-F1 {q.macro_f1.mean():.4f}"))
    lines += ["","## Matched model comparisons"]
    for left,right,title in [("ViT-ClosedFT","ViT Base","Closed FT vs Base"),("ViT-ClosedFT-SupCon","ViT-ClosedFT","Closed SupCon vs FT"),("ViT-FullFT","ViT-ClosedFT","Full vs Closed FT"),("ViT-FullFT-SupCon","ViT-FullFT","Full SupCon vs FT")]:
        lines.append(f"- {title}")
        for partition in ["closed","open"]:
            x=ok[(ok.model==left)&(ok.partition==partition)]; y=ok[(ok.model==right)&(ok.partition==partition)]; paired=x.merge(y,on=["prototype","distance","split"],suffixes=("_left","_right")); lines.append(f"  - {partition}: "+("unavailable" if paired.empty else f"accuracy {(paired.accuracy_left-paired.accuracy_right).mean():+.4f}, macro-F1 {(paired.macro_f1_left-paired.macro_f1_right).mean():+.4f}"))
    lines += ["","## Best settings"]
    for scope,partition in [("Overall",None),("Closed","closed"),("Open","open")]:
        q=ok if partition is None else ok[ok.partition==partition]
        for col in ["distance","prototype"]:
            score=q.groupby(col)[["accuracy","macro_f1"]].mean().sort_values(["accuracy","macro_f1"],ascending=False); lines.append(f"- {scope} best {col}: {score.index[0] if len(score) else 'unavailable'}")
    lines += ["","## Curve coverage"]
    for curve in curves:
        lines.append(f"- {curve['model']}: {curve['status']}"+(f" ({curve.get('reason','')})" if curve.get("reason") else ""))
    lines += ["",f"Completed result rows: {len(ok)}/120"]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n"); print(f"RESULT_ROWS={len(result)} OK_ROWS={len(ok)}")
    if len(result)!=120 or len(ok)!=120: raise RuntimeError(f"experiment incomplete: {len(ok)}/120 successful rows; partial artifacts preserved in {out}")

if __name__ == "__main__": main()

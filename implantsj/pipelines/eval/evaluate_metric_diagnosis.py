#!/usr/bin/env python3
"""Compact 42-label metric-representation diagnosis."""
import argparse, json, re, sys, unicodedata, zipfile
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
OPEN15 = {"simpleline", "exfeelinternal", "ssiii", "biohorizonexternal", "tsv", "scleantapered", "gsiii", "screwvent", "atlasinternal", "stella", "3i", "ankylos", "iu", "replaceselect", "core1"}
PASS = "PASS \u2014 checkpoint \uc5c6\uc74c"

def normalize_label_name(value):
    value = unicodedata.normalize("NFKC", str(value)).strip()
    value = re.sub(r"^\s*\d+\s*(?:[._:\-)]\s*)+", "", value)
    spaced = re.match(r"^\s*(\d+)\s+(.+)$", value)
    if spaced and int(spaced.group(1)) <= 42: value = spaced.group(2)
    value = re.sub(r"\s*\([^()]*\)\s*$", "", value)
    value=value.casefold().replace("biohorizons","biohorizon")
    return "".join(c for c in value if c.isalnum())

def match_label_name(value, normalized):
    key=normalize_label_name(value)
    if key in normalized: return normalized[key]
    candidates=[(len(candidate),label) for candidate,label in normalized.items() if (len(candidate)>=3 or candidate=="eb") and (candidate in key or key in candidate)]
    if not candidates: return ""
    best=max(length for length,_ in candidates); winners={label for length,label in candidates if length==best}
    if len(winners)!=1: raise RuntimeError(f"ambiguous English-name match for {value}: {sorted(winners)}")
    return winners.pop()

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

def manifest_rows(frame):
    return [(str(Path(row.path).resolve()),str(row.label),str(row.split)) for row in frame.itertuples()]

def extract_gt_zip(zip_path, base):
    zip_path=Path(zip_path); root=Path(base); root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        files=[info for info in archive.infolist() if not info.is_dir()]
        for info in files:
            target=(root/info.filename).resolve()
            if target!=root.resolve() and root.resolve() not in target.parents: raise RuntimeError(f"Unsafe zip: {info.filename}")
        if any(not (root/info.filename).is_file() or (root/info.filename).stat().st_size!=info.file_size for info in files): archive.extractall(root)
        bad=[info.filename for info in files if not (root/info.filename).is_file() or (root/info.filename).stat().st_size!=info.file_size]
    if bad: raise RuntimeError(f"GT extraction incomplete: {bad[:5]}")
    meta={"zip":str(zip_path.resolve()),"members":len(files),"image_members":sum(Path(info.filename).suffix.lower() in IMAGE_EXTENSIONS for info in files)}
    (root/"extraction.json").write_text(json.dumps(meta,indent=2)); return root,meta

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
        manifest_labels, manifest_splits, manifest_error, history_epochs, history_fields = [], {}, "", [], []
        if manifest.is_file():
            try:
                frame = load_manifest(manifest); manifest_labels = sorted(frame.label.astype(str).unique()); manifest_splits = frame.split.astype(str).value_counts().to_dict()
            except Exception as e: manifest_error = repr(e)
        history=path.parent.parent/"metrics/history.csv"
        if history.is_file():
            frame=pd.read_csv(history); history_fields=list(frame.columns); history_epochs=frame.epoch.astype(int).tolist() if "epoch" in frame else []
        return {"checkpoint":str(path.resolve()), "kind":"supcon" if any(k.startswith("projector.") for k in state) else "classifier",
                "model_name":args.get("model_name",""), "loss_name":args.get("loss_name",""), "labels":sorted(map(str,classes)),
                "image_size":int(args.get("image_size",224)),
                "manifest":str(manifest), "manifest_labels":manifest_labels, "manifest_splits":manifest_splits, "manifest_error":manifest_error, "epoch":ck.get("epoch",0),
                "history":str(history),"history_epochs":history_epochs,"history_fields":history_fields,"score":(ck.get("metrics",{}) or {}).get("macro_f1",-1), "readable":True}
    except Exception as e: return {"checkpoint":str(path), "readable":False, "error":repr(e)}

def candidate_facts(item, target, target_path):
    checkpoint, manifest = set(item.get("labels",[])), set(item.get("manifest_labels",[]))
    current=checkpoint==manifest==target and Path(item.get("manifest","")).resolve()==Path(target_path).resolve(); historical=bool()
    return current,historical,"current manifest exact"

def classifier_features(model, x):
    z = model.forward_features(x)
    if hasattr(model,"forward_head"):
        try: z = model.forward_head(z,pre_logits=True)
        except TypeError: z = model.forward_head(z)
    elif z.ndim == 3: z = z[:,0]
    return z if z.ndim == 2 else torch.nn.functional.adaptive_avg_pool2d(z,1).flatten(1)

def eval_transform(model, size, native):
    if native:
        from timm.data import create_transform, resolve_model_data_config
        config=resolve_model_data_config(model); transform=create_transform(**config,is_training=False)
        return transform,{"source":"timm pretrained config","input_size":config.get("input_size"),"mean":config.get("mean"),"std":config.get("std")}
    return build_transforms(size,False),{"source":"repository training transform","input_size":[3,size,size],"mean":[0.485,0.456,0.406],"std":[0.229,0.224,0.225]}

def make_model(item, device):
    kind = item["kind"]
    if kind == "pretrained_base":
        name, size = item["model_name"], int(item.get("image_size",224))
        model = create_classifier(name,0,True); transform,native=eval_transform(model,size,True); comparison,common=eval_transform(model,size,False); forward = lambda x:{"backbone":classifier_features(model,x)}
        return model.to(device).eval(),forward,transform,comparison,{"deployed":native,"common_backbone":common}
    path=Path(item["checkpoint"]); ck = torch.load(path,map_location="cpu"); args = ck.get("args",{}); state = clean(unwrap_state_dict(ck))
    name, size = args.get("model_name",item.get("model_name","vit_base_patch16_224")), int(args.get("image_size",224))
    if kind == "supcon":
        dim = int(args.get("embedding_dim") or state["projector.3.weight"].shape[0])
        model = MetricModel(name,dim,False,float(args.get("dropout",0)),float(args.get("drop_path_rate",0)))
        model.load_state_dict(state)
        def forward(x):
            backbone=model.forward_features(x); return {"backbone":backbone,"deployed":torch.nn.functional.normalize(model.projector(backbone),dim=1)}
    elif kind == "classifier":
        model = create_classifier(name,len(ck.get("class_to_idx",{})),False,float(args.get("dropout",0)),float(args.get("drop_path_rate",0)))
        model.load_state_dict(state); forward = lambda x:{"backbone":classifier_features(model,x)}
    else:
        model = create_classifier(name,0,False); loaded,_ = load_matching_weights(model,path)
        if loaded < int(len(model.state_dict())*.9): raise RuntimeError(f"Invalid Base checkpoint coverage ({loaded}/{len(model.state_dict())}): {path}")
        forward = lambda x:{"backbone":classifier_features(model,x)}
    if kind=="base":
        transform,native=eval_transform(model,size,True); comparison,common=eval_transform(model,size,False); return model.to(device).eval(),forward,transform,comparison,{"deployed":native,"common_backbone":common}
    transform,preprocess=eval_transform(model,size,False); return model.to(device).eval(),forward,transform,transform,{"deployed":preprocess,"common_backbone":preprocess}

@torch.no_grad()
def extract(model, forward, manifest, transform, batch, workers, device):
    df = load_manifest(manifest); labels = sorted(df.label.astype(str).unique()); mapping = {v:i for i,v in enumerate(labels)}
    ds = ImageClassificationDataset(manifest,None,mapping,transform,True)
    loader = DataLoader(ds,batch_size=batch,shuffle=False,num_workers=workers,pin_memory=device.type=="cuda")
    z,paths = {},[]
    for x,_,p in loader:
        for representation,value in forward(x.to(device)).items(): z.setdefault(representation,[]).append(value.cpu().numpy())
        paths += list(p)
    common={"labels":df.label.astype(str).to_numpy(object),"splits":df.split.astype(str).to_numpy(object),"paths":np.asarray(paths,dtype=object)}
    return {representation:{"features":np.concatenate(values),**common} for representation,values in z.items()}

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

def save_tsne(name, caches, out, seed, limit, representation="common_backbone"):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    rows = [(part,c["labels"][i],c["splits"][i],c["paths"][i],c["features"][i]) for part,c in caches.items() for i in range(len(c["labels"]))]
    total=len(rows); rows=stratified_rows(rows,limit,seed)
    x = np.stack([r[4] for r in rows]); x = PCA(min(50,x.shape[1],len(x)-1),random_state=seed).fit_transform(x) if x.shape[1]>50 else x
    xy = TSNE(2,perplexity=min(30,max(1,(len(x)-1)/3)),init="pca",learning_rate="auto",random_state=seed).fit_transform(x)
    df = pd.DataFrame({"partition":[r[0] for r in rows],"label":[r[1] for r in rows],"split":[r[2] for r in rows],"path":[r[3] for r in rows],"x":xy[:,0],"y":xy[:,1]})
    stem = name.replace(" ","_")+"_"+representation
    csv_path,npz_path,sampling_path=out/f"{stem}.csv",out/f"{stem}.npz",out/f"{stem}.sampling.json"
    train_test_png,valid_png=out/f"{stem}.png",out/f"{stem}_valid.png"
    df.to_csv(csv_path,index=False); np.savez_compressed(npz_path,features=x,coords=xy,**{c:df[c].to_numpy() for c in ["partition","label","split","path"]})
    xlim,ylim=axis_limits(df.x,False),axis_limits(df.y,False)
    counts={"/".join(map(str,k)):int(v) for k,v in df.groupby(["partition","split","label"]).size().items()}; meta={"representation":representation,"total_features":total,"sampled":len(df),"max_samples":limit,"seed":seed,"method":"deterministic round-robin by partition/split/label","partitions_overlaid":True,"axis_limits":{"x":xlim,"y":ylim},"counts":counts,"csv":str(csv_path),"npz":str(npz_path),"sampling":str(sampling_path),"train_test_png":str(train_test_png),"valid_png":str(valid_png)}; sampling_path.write_text(json.dumps(meta,indent=2))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    label_names=sorted(df.label.unique()); colors={label:plt.cm.turbo(i/max(1,len(label_names)-1)) for i,label in enumerate(label_names)}
    def panels(splits, path):
        fig,axes=plt.subplots(1,len(splits),figsize=(7*len(splits),6),squeeze=False)
        for ax,split in zip(axes[0],splits):
            q=df[df.split==split]
            for (part,label),g in q.groupby(["partition","label"],sort=True): ax.scatter(g.x,g.y,s=9,alpha=.65,color=colors[label],marker={"closed":"o","open":"^"}[part])
            ax.set_title(f"{name} [{representation.replace('_',' ')}]: closed + open / {split}"); ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_xticks([]); ax.set_yticks([])
        label_parts=df.drop_duplicates("label").set_index("label").partition
        handles=[Line2D([0],[0],marker={"closed":"o","open":"^"}[label_parts[label]],linestyle="",markersize=5,color=colors[label],label=label) for label in label_names]
        fig.legend(handles=handles,loc="center left",bbox_to_anchor=(.99,.5),fontsize=7); fig.tight_layout(rect=(0,0,.86,1)); fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)
    panels(["train","test"],train_test_png); panels(["valid"],valid_png)
    return meta

def save_prototype_tsne(name, caches, gtc, out, seed, count, representation="common_backbone"):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    rows=[]; coverage={}
    for part,c in caches.items():
        classes=sorted(set(c["labels"].tolist()))
        normalized={normalize_label_name(label):label for label in classes}
        if len(normalized)!=len(classes): raise RuntimeError(f"{part} labels collide after English-name normalization")
        for policy in PROTOS:
            support=gtc if policy=="gt" else c
            support_labels=np.asarray([match_label_name(label,normalized) for label in support["labels"]],dtype=object) if policy=="gt" else support["labels"]
            pool=np.flatnonzero(support_labels!="") if policy=="gt" else np.flatnonzero(np.isin(support_labels,classes)&(support["splits"]=="train"))
            missing=sorted(set(classes)-set(support_labels[pool].tolist()))
            coverage[f"{part}/{policy}"]={"available_labels":len(classes)-len(missing),"expected_labels":len(classes),"missing_labels":missing}
            if missing and policy!="gt": raise RuntimeError(f"{part}/{policy} prototype t-SNE coverage missing {missing}")
            if not len(pool): continue
            chosen=pool[choose_support_indices(support_labels[pool],support["paths"][pool],policy,count,seed)]
            prototypes,names=build_prototypes(support["features"],support_labels,chosen)
            rows += [(part,policy,label,int((support_labels[chosen]==label).sum()),vector) for label,vector in zip(names,prototypes)]
    if not any(row[1]=="gt" for row in rows): raise RuntimeError("GT English-name mapping matched zero labels")
    if len(rows)<3: raise RuntimeError("prototype t-SNE needs at least three available prototypes")
    features=np.stack([row[4] for row in rows]); reduced=PCA(min(50,features.shape[1],len(features)-1),random_state=seed).fit_transform(features) if features.shape[1]>50 else features
    coords=TSNE(2,perplexity=min(30,max(1,(len(reduced)-1)/3)),init="pca",learning_rate="auto",random_state=seed).fit_transform(reduced)
    df=pd.DataFrame({"model":name,"representation":representation,"partition":[row[0] for row in rows],"prototype":[row[1] for row in rows],"label":[row[2] for row in rows],"support_count":[row[3] for row in rows],"x":coords[:,0],"y":coords[:,1]})
    out.mkdir(parents=True,exist_ok=True); stem=name.replace(" ","_")+"_"+representation; csv_path,npz_path,png_path,meta_path=out/f"{stem}.csv",out/f"{stem}.npz",out/f"{stem}.png",out/f"{stem}.json"
    df.to_csv(csv_path,index=False); np.savez_compressed(npz_path,features=features,coords=coords,**{column:df[column].to_numpy() for column in ["partition","prototype","label","support_count"]})
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    styles={"gt":("#d62728","*","GT"),"mean":("#1f77b4","o","Mean"),"ss":("#2ca02c","^","SS")}; xlim,ylim=axis_limits(df.x,False),axis_limits(df.y,False)
    fig,ax=plt.subplots(figsize=(9,8))
    for _,group in df.groupby(["partition","label"],sort=True): ax.plot(group.x,group.y,color="0.75",linewidth=.5,zorder=1)
    for policy,(color,marker,label) in styles.items():
        group=df[df.prototype==policy]; ax.scatter(group.x,group.y,s=65 if policy=="gt" else 35,color=color,marker=marker,edgecolors="white",linewidths=.35,label=label,zorder=2)
    ax.set_title(f"{name} [{representation.replace('_',' ')}]: closed + open prototypes"); ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_xticks([]); ax.set_yticks([])
    handles=[Line2D([0],[0],marker=marker,linestyle="",markersize=8,color=color,label=label) for color,marker,label in styles.values()]
    fig.legend(handles=handles,loc="upper center",ncol=3); fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(png_path,dpi=180,bbox_inches="tight"); plt.close(fig)
    labels=sorted(df.label.unique()); colors={label:plt.cm.turbo(i/max(1,len(labels)-1)) for i,label in enumerate(labels)}; policy_pngs={}
    for policy,(_,marker,title) in styles.items():
        target=out/f"{stem}_{policy}.png"; fig,ax=plt.subplots(figsize=(9,8)); q=df[df.prototype==policy]
        for (part,label),group in q.groupby(["partition","label"],sort=True): ax.scatter(group.x,group.y,s=65 if policy=="gt" else 40,color=colors[label],marker={"closed":marker,"open":"X"}[part],edgecolors="white",linewidths=.35,label=label)
        ax.set_title(f"{name} [{representation.replace('_',' ')}]: closed + open / {title}"); ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_xticks([]); ax.set_yticks([])
        shown=q.drop_duplicates("label").set_index("label").partition; handles=[Line2D([0],[0],marker={"closed":marker,"open":"X"}[shown[label]],linestyle="",markersize=5,color=colors[label],label=label) for label in sorted(shown.index)]
        if handles: fig.legend(handles=handles,loc="center left",bbox_to_anchor=(.99,.5),fontsize=7)
        fig.tight_layout(rect=(0,0,.84,1)); fig.savefig(target,dpi=180,bbox_inches="tight"); plt.close(fig); policy_pngs[policy]=str(target)
    meta={"status":"ok","representation":representation,"seed":seed,"points":len(df),"partitions_overlaid":True,"csv":str(csv_path),"npz":str(npz_path),"png":str(png_path),"policy_pngs":policy_pngs,"coverage":coverage,"axis_limits":{"x":xlim,"y":ylim}}
    meta_path.write_text(json.dumps(meta,indent=2)); meta["metadata"]=str(meta_path); return meta

def axis_limits(values, bounded):
    low,high=float(values.min()),float(values.max()); span=high-low or max(abs(high)*.05,.02); pad=max(span*.12,.005 if bounded else span*.05)
    return (max(0.,low-pad),min(1.,high+pad)) if bounded else (low-pad,high+pad)

def save_curve(name, item, out):
    if item.get("kind") in {"base","pretrained_base"}: return {"model":name,"history":"","status":"N/A","png":"","csv":"","epochs":0}
    if not item.get("checkpoint"): return {"model":name,"history":"","status":"not_applicable","png":"","csv":"","epochs":0}
    history=Path(item["checkpoint"]).parent.parent/"metrics"/"history.csv"; stem=name.replace(" ","_")
    if not history.is_file(): return {"model":name,"history":str(history),"status":"missing","png":"","csv":"","epochs":0}
    df=pd.read_csv(history); copy=out/f"{stem}.csv"; df.to_csv(copy,index=False)
    groups=[("loss",["train_loss"]),("accuracy",[c for c in ["valid_accuracy","test_accuracy"] if c in df]),("macro-F1",[c for c in ["valid_macro_f1","test_macro_f1"] if c in df])]; groups=[g for g in groups if g[1] and all(c in df for c in g[1])]
    png=""
    if groups:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        x=df["epoch"] if "epoch" in df else np.arange(1,len(df)+1); fig,axes=plt.subplots(1,len(groups),figsize=(5*len(groups),4),squeeze=False)
        for ax,(title,columns) in zip(axes[0],groups):
            for column in columns: ax.plot(x,df[column],marker="o",label=column)
            values=pd.concat([df[column] for column in columns]).dropna()
            if len(values): ax.set_ylim(*axis_limits(values,title!="loss"))
            ax.set_title(title); ax.set_xlabel("epoch"); ax.legend()
        fig.tight_layout(); target=out/f"{stem}.png"; fig.savefig(target,dpi=180); plt.close(fig); png=str(target)
    missing=sorted({"test_accuracy","test_macro_f1"}-set(df.columns)); reasons=[]
    if len(df)<10: reasons.append(f"history has {len(df)}/10 epochs")
    if missing: reasons.append(f"history lacks per-epoch {', '.join(missing)}; available curves only")
    status="partial" if reasons else "ok"; reason="; ".join(reasons)
    ranking=[column for column in ["valid_macro_f1","valid_accuracy"] if column in df]; best=df.sort_values(ranking,ascending=False).iloc[0] if ranking else df.iloc[-1]
    result={"model":name,"history":str(history),"status":status,"csv":str(copy),"png":png,"reason":reason,"epochs":len(df),"best_epoch":int(best["epoch"]) if "epoch" in best else int(best.name)+1}
    for column in ["valid_accuracy","valid_macro_f1","test_accuracy","test_macro_f1"]: result[f"best_{column}"]=float(best[column]) if column in best and pd.notna(best[column]) else ""
    return result

def evaluate_caches(name, caches, gtc, args, representation, details=True):
    summary,predictions,members=[],[],[]
    for part,c in caches.items():
        classes=sorted(set(c["labels"].tolist())); label_to_idx={v:i for i,v in enumerate(classes)}
        for policy in PROTOS:
            support=gtc if policy=="gt" else c; required_split="gt" if policy=="gt" else "train"; pool=np.flatnonzero(np.isin(support["labels"],classes)&(support["splits"]==required_split))
            covered=set(support["labels"][pool].tolist()); missing=sorted(set(classes)-covered)
            if missing: raise RuntimeError(f"{part}/{policy} prototype coverage missing {missing}")
            chosen=pool[choose_support_indices(support["labels"][pool],support["paths"][pool],policy,args.ss_per_label,args.seed)]; prototypes,prototype_names=build_prototypes(support["features"],support["labels"],chosen)
            if set(prototype_names)!=set(classes): raise RuntimeError(f"{part}/{policy} prototype labels differ from query labels")
            if details: members += [{"model":name,"representation":representation,"partition":part,"prototype":policy,"label":support["labels"][i],"path":support["paths"][i]} for i in chosen]
            for distance in DISTS:
                for split in SPLITS:
                    qi=np.flatnonzero(c["splits"]==split)
                    if not len(qi): raise RuntimeError(f"{part}/{split} has no queries")
                    guess,score=prototype_predict(c["features"][qi],prototypes,prototype_names,distance); true=c["labels"][qi]; metrics=compute_metrics([label_to_idx[x] for x in true],[label_to_idx[x] for x in guess],classes)
                    summary.append({"model":name,"representation":representation,"prototype":policy,"distance":distance,"partition":part,"split":split,"status":"ok","reason":"",**{k:metrics[k] for k in ["accuracy","macro_f1","weighted_f1"]}})
                    if details: predictions += [{"model":name,"representation":representation,"prototype":policy,"distance":distance,"partition":part,"split":split,"path":c["paths"][j],"true_label":true[i],"pred_label":guess[i],"score":float(score[i]),"correct":bool(true[i]==guess[i])} for i,j in enumerate(qi)]
    if len(summary)!=24: raise RuntimeError(f"expected 24 result rows, got {len(summary)}")
    return summary,predictions,members

def save_partial(out, summary, backbone, predictions, members, curves, progress):
    (pd.DataFrame(summary) if summary else pd.DataFrame(columns=["model","prototype","distance","partition","split","status","reason"])).to_csv(out/"summary.csv",index=False)
    (pd.DataFrame(backbone) if backbone else pd.DataFrame(columns=["model","prototype","distance","partition","split","status","reason"])).to_csv(out/"backbone_summary.csv",index=False)
    (pd.DataFrame(predictions) if predictions else pd.DataFrame(columns=["model","prototype","distance","partition","split","path","true_label","pred_label","score","correct"])).to_csv(out/"predictions.csv",index=False)
    (pd.DataFrame(members) if members else pd.DataFrame(columns=["model","partition","prototype","label","path"])).to_csv(out/"prototype_membership.csv",index=False)
    (pd.DataFrame(curves) if curves else pd.DataFrame(columns=["model","history","status","csv","png","reason"])).to_csv(out/"curve_inventory.csv",index=False)
    (out/"progress.json").write_text(json.dumps(progress,indent=2))

def resume_tsne(out, seed, limit, ss_per_label):
    progress_path=out/"progress.json"; progress=json.loads(progress_path.read_text()); failures=[]
    for name in MODELS:
        if progress.get("models",{}).get(name,{}).get("status")==PASS:
            progress["models"][name]["tsne"]={"status":PASS}; progress["models"][name]["prototype_tsne"]={"status":PASS}; progress_path.write_text(json.dumps(progress,indent=2)); continue
        folder=out/"features"/name.replace(" ","_"); common={}; deployed={}
        try:
            def load(path):
                with np.load(path,allow_pickle=True) as archive: return {key:archive[key] for key in archive.files}
            for part in ["closed","open"]:
                path=folder/f"{part}_backbone.npz" if (folder/f"{part}_backbone.npz").is_file() else folder/f"{part}.npz"
                common[part]=load(path)
            common_gt=load(folder/("gt_backbone.npz" if (folder/"gt_backbone.npz").is_file() else "gt.npz"))
            common_meta=save_tsne(name,common,out/"tsne",seed,limit,"common_backbone")
            common_prototype=save_prototype_tsne(name,common,common_gt,out/"prototype_tsne",seed,ss_per_label,"common_backbone")
            if all((folder/f"{part}_backbone.npz").is_file() for part in ["closed","open"]):
                for part in ["closed","open"]: deployed[part]=load(folder/f"{part}.npz")
                deployed_meta={"status":"ok",**save_tsne(name,deployed,out/"tsne",seed,limit,"deployed")}
                deployed_prototype=save_prototype_tsne(name,deployed,load(folder/"gt.npz"),out/"prototype_tsne",seed,ss_per_label,"deployed")
            else: deployed_meta={"status":"same_as_common_backbone",**common_meta}; deployed_prototype={**common_prototype,"status":"same_as_common_backbone"}
            progress["models"].setdefault(name,{})["tsne"]={"status":"ok",**common_meta,"common_backbone":common_meta,"deployed":deployed_meta}
            progress["models"][name]["prototype_tsne"]={"status":"ok","common_backbone":common_prototype,"deployed":deployed_prototype}
        except Exception as error:
            progress["models"].setdefault(name,{})["tsne"]={"status":"unavailable","reason":repr(error)}; progress["models"][name]["prototype_tsne"]={"status":"unavailable","reason":repr(error)}; failures.append(name)
        progress_path.write_text(json.dumps(progress,indent=2))
    if failures: raise RuntimeError(f"t-SNE incomplete for {failures}; saved features can be resumed")
    return progress

def run_prototype_tsne_only(args, out):
    manifests={"closed":Path(args.closed_manifest),"open":Path(args.open_manifest)}; full_manifest=Path(args.full_manifest); gt_manifest=Path(args.gt_manifest or "")
    if not gt_manifest.is_file(): raise RuntimeError("--gt-manifest is required for --prototype-tsne-only")
    if args.ss_per_label<1: raise RuntimeError("--ss-per-label must be positive")
    manifest_labels={part:set(load_manifest(path).label.astype(str)) for part,path in manifests.items()}; full_labels=set(load_manifest(full_manifest).label.astype(str))
    if manifest_labels["closed"]&manifest_labels["open"] or manifest_labels["closed"]|manifest_labels["open"]!=full_labels: raise RuntimeError("closed/open/full manifest label sets are inconsistent")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type=="cpu" and not args.allow_cpu: raise RuntimeError("CUDA unavailable")
    checkpoints={"ViT Base":args.base_checkpoint,"ViT-ClosedFT":args.closed_checkpoint,"ViT-ClosedFT-SupCon":args.closed_supcon_checkpoint,"ViT-FullFT":args.full_checkpoint,"ViT-FullFT-SupCon":args.full_supcon_checkpoint}
    expected={"ViT-ClosedFT":("classifier",manifest_labels["closed"]),"ViT-ClosedFT-SupCon":("supcon",manifest_labels["closed"]),"ViT-FullFT":("classifier",full_labels),"ViT-FullFT-SupCon":("supcon",full_labels)}
    pass_status=globals().get("PASS","PASS checkpoint missing")
    progress={"mode":"prototype_tsne_only","gt_manifest":str(gt_manifest.resolve()),"seed":args.seed,"ss_per_label":args.ss_per_label,"models":{}}; failures=[]
    for name,raw_path in checkpoints.items():
        if not raw_path or not Path(raw_path).is_file(): progress["models"][name]={"status":pass_status,"reason":pass_status}; continue
        try:
            if name=="ViT Base": item={"kind":"base","checkpoint":str(Path(raw_path).resolve()),"model_name":args.base_model_name,"image_size":224}
            else:
                item=record(Path(raw_path))
                if not item.get("readable"): raise RuntimeError(item.get("error","checkpoint unreadable"))
                expected_kind,expected_labels=expected[name]
                if item["kind"]!=expected_kind: raise RuntimeError(f"{name} expects {expected_kind}, got {item['kind']}")
                if set(item.get("labels",[]))!=expected_labels: raise RuntimeError(f"{name} checkpoint labels do not match its {len(expected_labels)}-label manifest")
                stored_labels=set(item.get("manifest_labels",[]))
                if stored_labels and stored_labels!=expected_labels: raise RuntimeError(f"{name} stored training manifest labels do not match")
            model,forward,transform,comparison_transform,preprocess=make_model(item,device); deployed={}; common={}
            for part,manifest in manifests.items():
                representations=extract(model,forward,manifest,transform,args.batch_size,args.num_workers,device); deployed[part]=representations.get("deployed",representations["backbone"])
                common[part]=representations["backbone"] if comparison_transform is transform else extract(model,forward,manifest,comparison_transform,args.batch_size,args.num_workers,device)["backbone"]
            gt_representations=extract(model,forward,gt_manifest,transform,args.batch_size,args.num_workers,device); deployed_gt=gt_representations.get("deployed",gt_representations["backbone"])
            common_gt=gt_representations["backbone"] if comparison_transform is transform else extract(model,forward,gt_manifest,comparison_transform,args.batch_size,args.num_workers,device)["backbone"]
            common_meta=save_prototype_tsne(name,common,common_gt,out/"prototype_tsne",args.seed,args.ss_per_label,"common_backbone")
            if item["kind"]=="supcon" or comparison_transform is not transform: deployed_meta=save_prototype_tsne(name,deployed,deployed_gt,out/"prototype_tsne",args.seed,args.ss_per_label,"deployed")
            else: deployed_meta={**common_meta,"status":"same_as_common_backbone"}
            progress["models"][name]={"status":"ok","checkpoint":str(Path(raw_path).resolve()),"kind":item["kind"],"preprocessing":preprocess,"common_backbone":common_meta,"deployed":deployed_meta}
        except Exception as error:
            progress["models"][name]={"status":"unavailable","reason":repr(error)}; failures.append(name)
        finally:
            if "model" in locals(): del model
            if "forward" in locals(): del forward
            if "deployed" in locals(): del deployed
            if "common" in locals(): del common
            if "representations" in locals(): del representations
            if "gt_representations" in locals(): del gt_representations
            if "deployed_gt" in locals(): del deployed_gt
            if "common_gt" in locals(): del common_gt
            if device.type=="cuda": torch.cuda.empty_cache()
            (out/"progress.json").write_text(json.dumps(progress,indent=2))
    if failures: raise RuntimeError(f"prototype t-SNE incomplete for {failures}; partial artifacts preserved in {out}")
    print("PROTOTYPE_TSNE_OK")
    return progress

def validation_selected_results(ok):
    rows=[]; valid,test=ok[ok.split=="valid"],ok[ok.split=="test"]
    for (model,partition),group in valid.groupby(["model","partition"]):
        best=group.sort_values(["accuracy","macro_f1","prototype","distance"],ascending=[False,False,True,True]).iloc[0]
        held=test[(test.model==model)&(test.partition==partition)&(test.prototype==best.prototype)&(test.distance==best.distance)]
        if held.empty: continue
        held=held.iloc[0]; rows.append({"model":model,"partition":partition,"prototype":best.prototype,"distance":best.distance,"valid_accuracy":best.accuracy,"valid_macro_f1":best.macro_f1,"test_accuracy":held.accuracy,"test_macro_f1":held.macro_f1,"test_weighted_f1":held.weighted_f1})
    return pd.DataFrame(rows)

def write_report(out, args):
    result=pd.read_csv(out/"summary.csv"); ok=result[result.status=="ok"]; backbone_result=pd.read_csv(out/"backbone_summary.csv"); backbone_ok=backbone_result[backbone_result.status=="ok"]
    valid_ok,test_ok=ok[ok.split=="valid"],ok[ok.split=="test"]; backbone_test=backbone_ok[backbone_ok.split=="test"]; selected=json.loads((out/"checkpoint_selection.json").read_text()); progress=json.loads((out/"progress.json").read_text()); curves=pd.read_csv(out/"curve_inventory.csv").fillna("").to_dict("records")
    chosen=validation_selected_results(ok); chosen.to_csv(out/"selected_results.csv",index=False)
    tsne_note="all features" if args.tsne_max_samples<=0 else f"deterministic stratified sample, maximum {args.tsne_max_samples} per representation"
    def link(path,label):
        if not path: return ""
        path=Path(path)
        try: target=path.resolve().relative_to(out.resolve()).as_posix()
        except ValueError: target=str(path)
        return f"[{label}]({target})"
    def add_experiment(code,distance,names,title):
        lines.extend(["",f"## Experiment {code} \u2014 {distance.title()}: {title}","Held-out test accuracy / macro-F1 / weighted-F1 (deployed representation).","",f"| Set | Prototype | "+" | ".join(names)+" |",f"|---|---|"+"---|"*len(names)])
        for partition in ["closed","open"]:
            for prototype,label in [("gt","GT"),("mean","Mean"),("ss","SS")]:
                cells=[]
                for name in names:
                    q=test_ok[(test_ok.model==name)&(test_ok.partition==partition)&(test_ok.prototype==prototype)&(test_ok.distance==distance)]
                    cells.append((PASS if name not in selected else "unavailable") if q.empty else f"{q.iloc[0].accuracy:.4f} / {q.iloc[0].macro_f1:.4f} / {q.iloc[0].weighted_f1:.4f}")
                lines.append(f"| {partition.title()} | {label} | "+" | ".join(cells)+" |")
    def delta(left,right,table):
        if left not in selected or right not in selected: return f"{PASS} ({left if left not in selected else right})"
        values=[]
        for partition in ["closed","open"]:
            x=table[(table.model==left)&(table.partition==partition)]; y=table[(table.model==right)&(table.partition==partition)]; paired=x.merge(y,on=["prototype","distance"],suffixes=("_left","_right"))
            if len(paired):
                da,df=(paired.accuracy_left-paired.accuracy_right).mean(),(paired.macro_f1_left-paired.macro_f1_right).mean(); verdict="improved" if da>0 and df>0 else "degraded" if da<0 and df<0 else "mixed/no material change"
                values.append(f"{partition} {verdict}: \u0394accuracy {da:+.4f}, \u0394macro-F1 {df:+.4f}")
            else: values.append(f"{partition} unavailable")
        return "; ".join(values)
    def setting_result(column):
        validation=valid_ok.groupby(column)[["accuracy","macro_f1"]].mean().sort_values(["accuracy","macro_f1"],ascending=False)
        heldout=test_ok.groupby(column)[["accuracy","macro_f1"]].mean()
        if len(validation)<2 or not set(validation.index[:2])<=set(heldout.index): return "unavailable"
        winner,runner=validation.index[:2]; return f"{winner}; held-out test accuracy {heldout.loc[winner].accuracy:.4f}, macro-F1 {heldout.loc[winner].macro_f1:.4f}; vs {runner} \u0394accuracy {heldout.loc[winner].accuracy-heldout.loc[runner].accuracy:+.4f}, \u0394macro-F1 {heldout.loc[winner].macro_f1-heldout.loc[runner].macro_f1:+.4f}"
    lines=["# Metric Learning Diagnosis","",f"Artifacts: `{out.resolve()}`",f"Seed: {args.seed}; SS: {args.ss_per_label}; GT unmatched: 0",f"t-SNE: {tsne_note}; common-backbone plots cover all available models, and deployed plots additionally cover representations that differ (Base preprocessing and SupCon projector). Full features remain in NPZ.","Preprocessing: ViT Base deployed results use its timm pretrained configuration; all common-backbone comparisons use the repository transform. Fine-tuned deployed results use that same training transform. Exact values are in progress.json.","SupCon deployed results use the model's L2-normalized projector output. Common-backbone results remove projector dimension/normalization confounding, but remain a training-configuration comparison because CE and SupCon runs also differ in batch size, LR, and augmentation.","","## Artifact index","- "+" \u00b7 ".join([link(out/"summary.csv","deployed metrics"),link(out/"backbone_summary.csv","common-backbone metrics"),link(out/"selected_results.csv","validation-selected test results"),link(out/"predictions.csv","predictions")]),"- "+" \u00b7 ".join([link(out/"curve_inventory.csv","curve inventory"),link(out/"prototype_membership.csv","prototype membership"),link(out/"checkpoint_selection.json","checkpoint selection"),link(out/"checkpoint_inventory.csv","checkpoint inventory")])]
    add_experiment("1a","euclidean",["ViT Base","ViT-ClosedFT","ViT-ClosedFT-SupCon"],"Base vs ClosedFT vs ClosedFT-SupCon")
    add_experiment("1b","cosine",["ViT Base","ViT-ClosedFT","ViT-ClosedFT-SupCon"],"Base vs ClosedFT vs ClosedFT-SupCon")
    add_experiment("2a","euclidean",["ViT-ClosedFT","ViT-FullFT"],"ClosedFT vs FullFT")
    add_experiment("2b","cosine",["ViT-ClosedFT","ViT-FullFT"],"ClosedFT vs FullFT")
    add_experiment("4a","euclidean",["ViT-ClosedFT","ViT-ClosedFT-SupCon"],"ClosedFT vs ClosedFT-SupCon")
    add_experiment("4b","cosine",["ViT-ClosedFT","ViT-ClosedFT-SupCon"],"ClosedFT vs ClosedFT-SupCon")
    add_experiment("5a","euclidean",["ViT-FullFT","ViT-FullFT-SupCon"],"FullFT vs FullFT-SupCon")
    add_experiment("5b","cosine",["ViT-FullFT","ViT-FullFT-SupCon"],"FullFT vs FullFT-SupCon")
    lines += ["","## Five research-question conclusions","Common-backbone deltas below are matched held-out-test marginal means across GT/Mean/SS \u00d7 Euclidean/cosine; positive favors the first model.",f"1. Closed-set fine-tuning vs general Base features: {delta('ViT-ClosedFT','ViT Base',backbone_test)}.",f"2. SupCon effect: ClosedFT-SupCon vs ClosedFT \u2014 {delta('ViT-ClosedFT-SupCon','ViT-ClosedFT',backbone_test)}; FullFT-SupCon vs FullFT \u2014 {delta('ViT-FullFT-SupCon','ViT-FullFT',backbone_test)}.",f"3. Full-set training vs closed-only training: {delta('ViT-FullFT','ViT-ClosedFT',backbone_test)}.",f"4. Best distance selected by validation: {setting_result('distance')}.",f"5. Best prototype selected by validation: {setting_result('prototype')}." ]
    lines += ["","## Validation-selected joint setting, held-out test"]
    for row in chosen.itertuples(): lines.append(f"- {row.model} / {row.partition}: {row.prototype} + {row.distance}; validation accuracy {row.valid_accuracy:.4f}, macro-F1 {row.valid_macro_f1:.4f}; test accuracy {row.test_accuracy:.4f}, macro-F1 {row.test_macro_f1:.4f}")
    lines += ["","## Held-out test performance by partition","Values below are marginal means across all six prototype \u00d7 distance settings."]
    for partition in ["closed","open"]:
        lines += ["",f"### {partition.title()}"]
        for name in MODELS:
            q=test_ok[(test_ok.model==name)&(test_ok.partition==partition)]; lines.append(f"- {name}: "+((PASS if name not in selected else "unavailable") if q.empty else f"accuracy {q.accuracy.mean():.4f}, macro-F1 {q.macro_f1.mean():.4f}"))
    lines += ["","## Matched held-out test comparisons","Deltas are marginal means across matched prototype \u00d7 distance settings."]
    for left,right,title,table in [("ViT-ClosedFT","ViT Base","Closed FT vs Base, model-native preprocessing",test_ok),("ViT-ClosedFT","ViT Base","Closed FT vs Base, common preprocessing/backbone",backbone_test),("ViT-ClosedFT-SupCon","ViT-ClosedFT","Closed SupCon backbone vs FT backbone",backbone_test),("ViT-FullFT","ViT-ClosedFT","Full vs Closed FT",test_ok),("ViT-FullFT-SupCon","ViT-FullFT","Full SupCon backbone vs FT backbone",backbone_test)]:
        lines.append(f"- {title}")
        for partition in ["closed","open"]:
            x=table[(table.model==left)&(table.partition==partition)]; y=table[(table.model==right)&(table.partition==partition)]; paired=x.merge(y,on=["prototype","distance"],suffixes=("_left","_right")); lines.append(f"  - {partition}: "+("unavailable" if paired.empty else f"accuracy {(paired.accuracy_left-paired.accuracy_right).mean():+.4f}, macro-F1 {(paired.macro_f1_left-paired.macro_f1_right).mean():+.4f}"))
    lines += ["","## SupCon projector impact on held-out test"]
    for name in ["ViT-ClosedFT-SupCon","ViT-FullFT-SupCon"]:
        lines.append(f"- {name} projected-normalized vs backbone")
        for partition in ["closed","open"]:
            x=test_ok[(test_ok.model==name)&(test_ok.partition==partition)]; y=backbone_test[(backbone_test.model==name)&(backbone_test.partition==partition)]; paired=x.merge(y,on=["prototype","distance"],suffixes=("_projected","_backbone")); lines.append(f"  - {partition}: "+("unavailable" if paired.empty else f"accuracy {(paired.accuracy_projected-paired.accuracy_backbone).mean():+.4f}, macro-F1 {(paired.macro_f1_projected-paired.macro_f1_backbone).mean():+.4f}"))
    lines += ["","## Marginal settings selected on validation, reported on test"]
    for scope,partition in [("Overall",None),("Closed","closed"),("Open","open")]:
        validation=valid_ok if partition is None else valid_ok[valid_ok.partition==partition]; heldout=test_ok if partition is None else test_ok[test_ok.partition==partition]
        for col in ["distance","prototype"]:
            score=validation.groupby(col)[["accuracy","macro_f1"]].mean().sort_values(["accuracy","macro_f1"],ascending=False); winner=score.index[0] if len(score) else None; test_score=heldout[heldout[col]==winner][["accuracy","macro_f1"]].mean() if winner else None; lines.append(f"- {scope} best {col}: "+("unavailable" if winner is None else f"{winner} (validation accuracy {score.iloc[0].accuracy:.4f}; test accuracy {test_score.accuracy:.4f}, macro-F1 {test_score.macro_f1:.4f})"))
    lines += ["","## Training curves"]
    for curve in curves:
        details=f"{curve['status']}; {int(curve.get('epochs') or 0)} epochs"
        if curve.get("best_epoch") not in (None,""): details+=f"; best validation epoch {int(curve['best_epoch'])}"
        for label,column in [("valid acc","best_valid_accuracy"),("valid F1","best_valid_macro_f1"),("test acc","best_test_accuracy"),("test F1","best_test_macro_f1")]:
            if curve.get(column) not in (None,""): details+=f"; {label} {float(curve[column]):.4f}"
        assets=" \u00b7 ".join(x for x in [link(curve.get("png"),"curve"),link(curve.get("csv"),"history")] if x)
        lines.append(f"- {curve['model']}: {details}"+(f"; {assets}" if assets else "")+(f" ({curve['reason']})" if curve.get("reason") else ""))
    lines += ["","## t-SNE coverage"]
    for name in MODELS:
        info=progress["models"].get(name,{}).get("tsne",{}); common=info.get("common_backbone",{}); deployed=info.get("deployed",{})
        assets=[link(common.get("train_test_png"),"common train/test"),link(common.get("valid_png"),"common valid")]
        if deployed.get("status")!="same_as_common_backbone": assets += [link(deployed.get("train_test_png"),"deployed train/test"),link(deployed.get("valid_png"),"deployed valid")]
        prototype=progress["models"].get(name,{}).get("prototype_tsne",{}); prototype_common=prototype.get("common_backbone",{}); prototype_deployed=prototype.get("deployed",{})
        assets.append(link(prototype_common.get("png"),"common GT/Mean/SS prototypes"))
        if prototype_deployed.get("status")!="same_as_common_backbone": assets.append(link(prototype_deployed.get("png"),"deployed GT/Mean/SS prototypes"))
        lines.append(f"- {name}: {info.get('status','unavailable')}"+("; "+" \u00b7 ".join(x for x in assets if x) if any(assets) else ""))
    result_covered=result.status.isin(["ok",PASS]); backbone_covered=backbone_result.status.isin(["ok",PASS])
    lines += ["",f"Covered deployed result rows: {int(result_covered.sum())}/120 (OK {len(ok)}, PASS {int((result.status==PASS).sum())})",f"Covered common-backbone rows: {int(backbone_covered.sum())}/120 (OK {len(backbone_ok)}, PASS {int((backbone_result.status==PASS).sum())})"]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n"); print(f"RESULT_ROWS={len(result)} OK_ROWS={len(ok)} BACKBONE_OK_ROWS={len(backbone_ok)}")
    return len(result)==len(backbone_result)==120 and result_covered.all() and backbone_covered.all(),all(progress["models"].get(name,{}).get(kind,{}).get("status")=="ok" for name in selected for kind in ["tsne","prototype_tsne"]),len(ok),len(backbone_ok)

def run_full_only(args, out):
    manifest=Path(args.full_manifest); frame=load_manifest(manifest)
    if "split" not in frame: raise RuntimeError("full manifest has no split column")
    classes=sorted(frame.label.astype(str).unique())
    missing={label:sorted({"train","test"}-set(group.split.astype(str))) for label,group in frame.groupby("label")}; missing={label:splits for label,splits in missing.items() if splits}
    errors=[]
    if len(classes)!=42: errors.append(f"full label count is {len(classes)}, not 42")
    if missing: errors.append(f"labels missing train/test splits: {missing}")
    if errors: raise RuntimeError(errors)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type=="cpu" and not args.allow_cpu: raise RuntimeError("CUDA unavailable")
    item={"kind":"pretrained_base","model_name":args.base_model_name,"image_size":224}
    model,forward,transform,_,preprocess=make_model(item,device)
    cache=extract(model,forward,manifest,transform,args.batch_size,args.num_workers,device)["backbone"]
    train_idx=np.flatnonzero(cache["splits"]=="train"); test_idx=np.flatnonzero(cache["splits"]=="test")
    prototypes,prototype_names=build_prototypes(cache["features"],cache["labels"],train_idx)
    if prototype_names!=classes: raise RuntimeError("train mean-prototype labels differ from the 42 manifest labels")
    label_to_idx={label:i for i,label in enumerate(classes)}; true_labels=cache["labels"][test_idx]
    summary,predictions,metric_results=[],[],{}
    for distance in DISTS:
        predicted,scores=prototype_predict(cache["features"][test_idx],prototypes,prototype_names,distance)
        metrics=compute_metrics([label_to_idx[label] for label in true_labels],[label_to_idx[label] for label in predicted],classes)
        summary.append({"model":"ViT Base","model_name":args.base_model_name,"representation":"backbone","prototype":"mean","distance":distance,"partition":"full","split":"test","status":"ok",**{key:metrics[key] for key in ["accuracy","macro_f1","weighted_f1","macro_precision","macro_recall"]}})
        metric_results[distance]=metrics
        predictions += [{"model":"ViT Base","model_name":args.base_model_name,"representation":"backbone","prototype":"mean","distance":distance,"partition":"full","split":"test","path":str(cache["paths"][index]),"true_label":str(true_labels[position]),"pred_label":str(predicted[position]),"score":float(scores[position]),"correct":bool(predicted[position]==true_labels[position])} for position,index in enumerate(test_idx)]
    pd.DataFrame(summary).to_csv(out/"summary.csv",index=False); pd.DataFrame(predictions).to_csv(out/"predictions.csv",index=False)
    payload={"model":"ViT Base","model_name":args.base_model_name,"source":"timm pretrained weights","checkpoint":None,"manifest":str(manifest.resolve()),"label_count":len(classes),"train_samples":len(train_idx),"test_samples":len(test_idx),"prototype":"train mean","preprocessing":preprocess,"results":metric_results}
    (out/"metrics.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    for row in summary: print(f"{row['distance']}: accuracy={row['accuracy']:.6f}, macro_f1={row['macro_f1']:.6f}")
    print(f"FULL_ONLY_RESULTS={out/'summary.csv'}")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--closed-manifest"); p.add_argument("--open-manifest"); p.add_argument("--full-manifest",required=True)
    p.add_argument("--checkpoint-root",action="append",default=[]); p.add_argument("--base-checkpoint"); p.add_argument("--base-model-name",default="vit_large_patch16_224.augreg_in21k_ft_in1k")
    p.add_argument("--closed-checkpoint"); p.add_argument("--closed-supcon-checkpoint"); p.add_argument("--full-checkpoint"); p.add_argument("--full-supcon-checkpoint")
    p.add_argument("--gt-zip"); p.add_argument("--gt-root"); p.add_argument("--gt-manifest"); p.add_argument("--output-dir",required=True)
    p.add_argument("--ss-per-label",type=int,default=5); p.add_argument("--seed",type=int,default=42); p.add_argument("--batch-size",type=int,default=64); p.add_argument("--num-workers",type=int,default=8); p.add_argument("--tsne-max-samples",type=int,default=0); p.add_argument("--allow-cpu",action="store_true"); p.add_argument("--inventory-only",action="store_true"); p.add_argument("--resume-tsne",action="store_true"); p.add_argument("--prototype-tsne-only",action="store_true"); p.add_argument("--full-only",action="store_true"); a=p.parse_args()
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    if a.full_only: run_full_only(a,out); return
    if not a.closed_manifest or not a.open_manifest or not a.checkpoint_root: p.error("--closed-manifest, --open-manifest, and --checkpoint-root are required unless --full-only is used")
    (out/"completion.json").unlink(missing_ok=True); (out/"features").mkdir(exist_ok=True); (out/"tsne").mkdir(exist_ok=True); (out/"prototype_tsne").mkdir(exist_ok=True); (out/"curves").mkdir(exist_ok=True)
    if a.prototype_tsne_only: run_prototype_tsne_only(a,out); return
    if a.resume_tsne:
        resume_tsne(out,a.seed,a.tsne_max_samples,a.ss_per_label); numeric_ok,visual_ok,ok_rows,backbone_rows=write_report(out,a)
        if not numeric_ok or not visual_ok: raise RuntimeError(f"experiment incomplete after t-SNE resume: deployed {ok_rows}/120, backbone {backbone_rows}/120")
        print("TSNE_RESUME_OK"); return
    manifests={"closed":Path(a.closed_manifest),"open":Path(a.open_manifest),"full":Path(a.full_manifest)}; frames={k:load_manifest(v) for k,v in manifests.items()}; labels={k:set(v.label.astype(str)) for k,v in frames.items()}; rows={k:manifest_rows(v) if "split" in v else [] for k,v in frames.items()}
    errors=[]; split_errors={}
    if len(labels["closed"])!=27: errors.append("closed label count is not 27")
    if len(labels["open"])!=15: errors.append("open label count is not 15")
    if len(labels["full"])!=42: errors.append("full label count is not 42")
    if labels["closed"]&labels["open"]: errors.append("closed/open overlap")
    if labels["closed"]|labels["open"]!=labels["full"]: errors.append("closed/open union differs from full")
    if {normalize_label_name(label) for label in labels["open"]}!=OPEN15: errors.append("open labels do not match the required Open15 English names")
    normalized_closed={normalize_label_name(label) for label in labels["closed"]}
    if "tsiii" not in normalized_closed or "ssiii" in normalized_closed: errors.append("closed labels must contain TS III and exclude SS III")
    for part,frame in frames.items():
        if "split" not in frame: errors.append(f"{part} manifest has no split column"); continue
        missing={label:sorted(set(["train","valid","test"])-set(group.split.astype(str))) for label,group in frame.groupby("label")}; missing={k:v for k,v in missing.items() if v}
        if missing: split_errors[part]=missing; errors.append(f"{part} labels missing train/valid/test splits")
        if len(set(rows[part]))!=len(rows[part]): errors.append(f"{part} manifest has duplicate path/label/split rows")
        paths=[row[0] for row in rows[part]]
        if len(set(paths))!=len(paths): errors.append(f"{part} manifest reuses a path")
    row_union_exact=set(rows["full"])==set(rows["closed"])|set(rows["open"]) and not set(rows["closed"])&set(rows["open"])
    if not row_union_exact: errors.append("full manifest rows are not the exact disjoint union of closed and open")
    audit={"ok":not errors,"errors":errors,"split_errors":split_errors,"row_union_exact":row_union_exact,"manifests":{k:{"path":str(v.resolve()),"rows":len(frames[k]),"labels":len(labels[k]),"splits":frames[k].split.value_counts().to_dict() if "split" in frames[k] else {}} for k,v in manifests.items()}}; (out/"manifest_audit.json").write_text(json.dumps(audit,indent=2))
    if errors: raise RuntimeError(errors)
    paths=set()
    for root in map(Path,a.checkpoint_root): paths.update(root.rglob("best.pt")); paths.update(root.rglob("best*.pth"))
    explicit={"ViT-ClosedFT":a.closed_checkpoint,"ViT-ClosedFT-SupCon":a.closed_supcon_checkpoint,"ViT-FullFT":a.full_checkpoint,"ViT-FullFT-SupCon":a.full_supcon_checkpoint}
    paths.update(Path(path) for path in explicit.values() if path and Path(path).is_file())
    records=[record(x) for x in sorted(paths)]; pd.DataFrame(records).to_csv(out/"checkpoint_inventory.csv",index=False); selected={}; selection_errors={}; missing_models=set(); unreadable_seen=any(not item.get("readable") for item in records)
    needs={"ViT-ClosedFT":("classifier","closed"),"ViT-ClosedFT-SupCon":("supcon","closed"),"ViT-FullFT":("classifier","full"),"ViT-FullFT-SupCon":("supcon","full")}
    for name,(kind,part) in needs.items():
        candidates=[]; target_seen=False
        for r in records:
            eligible,own,basis=candidate_facts(r,labels[part],manifests[part])
            target_seen = target_seen or bool(r.get("readable") and r.get("kind")==kind and set(r.get("labels",[]))==labels[part])
            if r.get("readable") and r.get("kind")==kind and r.get("model_name")==a.base_model_name and r.get("image_size")==224 and (kind!="supcon" or r.get("loss_name") in ["","supcon"]) and eligible: candidates.append((r,own,basis))
        if explicit[name]:
            wanted_path=Path(explicit[name]); wanted=str(wanted_path.resolve())
            if not wanted_path.is_file():
                if candidates: selection_errors[name]=f"declared checkpoint absent; using discovered exact-{part}-manifest checkpoint"; explicit[name]=None
                elif target_seen or unreadable_seen: selection_errors[name]="declared checkpoint absent but existing checkpoint inventory is invalid or ambiguous"; continue
                else: selection_errors[name]=PASS; missing_models.add(name); continue
            else:
                matches=[x for x in candidates if str(Path(x[0]["checkpoint"]).resolve())==wanted]
                if matches: candidates=matches
                else: selection_errors[name]=f"explicit checkpoint failed exact {part} manifest/model validation: {wanted}"; candidates=[]
        if candidates:
            chosen,own,basis=max(candidates,key=lambda x:(x[1],set(["train","valid","test"])<=set(x[0]["manifest_splits"]),x[0]["epoch"],x[0]["score"]))
            selected[name]={**chosen,"selection_basis":("explicit " if explicit[name] else "")+basis,"missing_target_labels":[],"own_manifest_backed":own,"own_manifest_complete_splits":set(["train","valid","test"])<=set(chosen["manifest_splits"])}
        elif not explicit[name] and target_seen: selection_errors[name]=f"existing exact-{part}-manifest {kind} checkpoint failed model validation"
        elif not explicit[name] and unreadable_seen: selection_errors[name]="unreadable existing checkpoint prevents proving target absence"
        elif not explicit[name]: selection_errors[name]=PASS; missing_models.add(name)
    selected["ViT Base"]={"kind":"base","checkpoint":str(Path(a.base_checkpoint).resolve()),"model_name":a.base_model_name,"selection_basis":"explicit base checkpoint"} if a.base_checkpoint else {"kind":"pretrained_base","model_name":a.base_model_name,"image_size":224,"selection_basis":"timm pretrained weights"}
    (out/"checkpoint_selection.json").write_text(json.dumps(selected,indent=2)); (out/"checkpoint_selection_errors.json").write_text(json.dumps(selection_errors,indent=2)); print("SELECTED=",",".join(x for x in MODELS if x in selected)); print("MISSING=",",".join(x for x in MODELS if x not in selected))
    if a.inventory_only: return
    if not a.gt_zip or not a.gt_root: raise RuntimeError("GT paths required")
    if a.ss_per_label < 1: raise RuntimeError("ss-per-label must be positive")
    gtroot,gt_extract=extract_gt_zip(a.gt_zip,a.gt_root)
    gt,_=map_gt_label_folders(gtroot,labels["full"],out/"gt_mapping.csv"); gtmanifest=out/"gt_manifest.csv"; pd.DataFrame([{"path":str(p),"label":l,"split":"gt"} for l,ps in gt.items() for p in ps]).to_csv(gtmanifest,index=False)
    if set(gt)!=labels["full"]: raise RuntimeError("GT prototype coverage differs from full labels")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type=="cpu" and not a.allow_cpu: raise RuntimeError("CUDA unavailable")
    summary,backbone_summary,predictions,members,curves=[],[],[],[],[]; progress={"expected_result_rows":120,"expected_backbone_rows":120,"tsne_max_samples":a.tsne_max_samples,"models":{}}
    for name in MODELS:
        if name in missing_models:
            passed=[{"model":name,"representation":"unavailable","prototype":p,"distance":d,"partition":q,"split":s,"status":PASS,"reason":PASS} for p in PROTOS for d in DISTS for q in ["closed","open"] for s in SPLITS]
            summary+=passed; backbone_summary += [{**row,"representation":"backbone"} for row in passed]; curves.append({"model":name,"history":"","status":PASS,"png":"","csv":"","reason":PASS,"epochs":0}); progress["models"][name]={"status":PASS,"reason":PASS,"result_rows":24,"backbone_rows":24,"tsne":{"status":PASS},"prototype_tsne":{"status":PASS}}; save_partial(out,summary,backbone_summary,predictions,members,curves,progress); continue
        reason=""
        try:
            if name not in selected: raise RuntimeError(selection_errors.get(name,"checkpoint missing"))
            item=selected[name]
            try: curves.append(save_curve(name,item,out/"curves"))
            except Exception as curve_error: curves.append({"model":name,"history":"","status":"unavailable","png":"","reason":repr(curve_error)})
            model,forward,transform,comparison_transform,preprocess=make_model(item,device); caches={}; backbone_caches={}; folder=out/"features"/name.replace(" ","_"); folder.mkdir(exist_ok=True)
            for part in ["closed","open"]:
                representations=extract(model,forward,manifests[part],transform,a.batch_size,a.num_workers,device); caches[part]=representations.get("deployed",representations["backbone"])
                comparison=representations if comparison_transform is transform else extract(model,forward,manifests[part],comparison_transform,a.batch_size,a.num_workers,device); backbone_caches[part]=comparison["backbone"]; np.savez_compressed(folder/f"{part}.npz",**caches[part])
                if "deployed" in representations or comparison_transform is not transform: np.savez_compressed(folder/f"{part}_backbone.npz",**backbone_caches[part])
            gt_representations=extract(model,forward,gtmanifest,transform,a.batch_size,a.num_workers,device); gtc=gt_representations.get("deployed",gt_representations["backbone"]); gt_comparison=gt_representations if comparison_transform is transform else extract(model,forward,gtmanifest,comparison_transform,a.batch_size,a.num_workers,device); gt_backbone=gt_comparison["backbone"]; np.savez_compressed(folder/"gt.npz",**gtc)
            if "deployed" in gt_representations or comparison_transform is not transform: np.savez_compressed(folder/"gt_backbone.npz",**gt_backbone)
            representation="projected_normalized" if item["kind"]=="supcon" else "backbone"
            local_summary,local_predictions,local_members=evaluate_caches(name,caches,gtc,a,representation,True)
            if item["kind"]=="supcon" or comparison_transform is not transform: local_backbone,_,_=evaluate_caches(name,backbone_caches,gt_backbone,a,"backbone",False)
            else: local_backbone=[{**row,"representation":"backbone"} for row in local_summary]
            summary+=local_summary; backbone_summary+=local_backbone; predictions+=local_predictions; members+=local_members
            progress["models"][name]={"status":"ok","result_rows":24,"backbone_rows":24,"representation":representation,"preprocessing":preprocess,"tsne":{"status":"pending","representation":"common_backbone"},"prototype_tsne":{"status":"pending"}}
        except Exception as e:
            reason=repr(e); unavailable=[{"model":name,"representation":"unavailable","prototype":p,"distance":d,"partition":q,"split":s,"status":"unavailable","reason":reason} for p in PROTOS for d in DISTS for q in ["closed","open"] for s in SPLITS]; summary+=unavailable; backbone_summary += [{**row,"representation":"backbone"} for row in unavailable]; progress["models"][name]={"status":"unavailable","reason":reason,"result_rows":24,"backbone_rows":24}
            if not any(r["model"]==name for r in curves): curves.append({"model":name,"history":"","status":"unavailable","png":"","reason":reason})
        finally:
            if "forward" in locals(): del forward
            if "model" in locals(): del model
            if "caches" in locals(): del caches
            if "backbone_caches" in locals(): del backbone_caches
            if "representations" in locals(): del representations
            if "comparison" in locals(): del comparison
            if "gt_representations" in locals(): del gt_representations
            if "gt_comparison" in locals(): del gt_comparison
            if "gtc" in locals(): del gtc
            if "gt_backbone" in locals(): del gt_backbone
            if device.type=="cuda": torch.cuda.empty_cache()
            save_partial(out,summary,backbone_summary,predictions,members,curves,progress)
    try: progress=resume_tsne(out,a.seed,a.tsne_max_samples,a.ss_per_label)
    except RuntimeError: progress=json.loads((out/"progress.json").read_text())
    numeric_ok,visual_ok,ok_rows,backbone_rows=write_report(out,a)
    if not numeric_ok or not visual_ok: raise RuntimeError(f"experiment incomplete: deployed {ok_rows}/120, backbone {backbone_rows}/120, t-SNE complete={visual_ok}; partial artifacts preserved in {out}")

if __name__ == "__main__": main()

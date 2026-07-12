#!/usr/bin/env python3
"""Compact 42-label metric-representation diagnosis."""
import argparse, hashlib, importlib.metadata, json, platform, re, sys, unicodedata, zipfile
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

def run_fingerprint(checkpoints, manifests, gt_zip, config, base_checkpoint=None):
    def facts(path):
        path=Path(path).resolve(); return {"path":str(path),"sha256":digest(path),"bytes":path.stat().st_size}
    if base_checkpoint: base=facts(base_checkpoint)
    else:
        import timm
        base={"kind":"implicit_timm_pretrained","timm_version":timm.__version__,"pretrained_cfg":json.loads(json.dumps(timm.get_pretrained_cfg(config["base_model_name"]).to_dict(),default=str))}
    sources={str(path.relative_to(ROOT)):facts(path) for path in [Path(__file__),ROOT/"src/data.py",ROOT/"src/models.py",ROOT/"src/metrics.py"]}
    packages={"torch":"torch","torchvision":"torchvision","timm":"timm","sklearn":"scikit-learn","numpy":"numpy","pandas":"pandas","matplotlib":"matplotlib","pillow":"Pillow"}
    runtime={"python":platform.python_version(),**{name:importlib.metadata.version(package) for name,package in packages.items()}}
    return {"schema":2,"evaluator":facts(__file__),"sources":sources,"runtime":runtime,"base":base,"checkpoints":{name:facts(path) for name,path in sorted(checkpoints.items())},"manifests":{name:facts(path) for name,path in sorted(manifests.items())},"gt_zip":facts(gt_zip),"config":config}

def manifest_rows(frame):
    return [(str(Path(row.path).resolve()),str(row.label),str(row.split)) for row in frame.itertuples()]

def extract_gt_zip(zip_path, base):
    zip_path=Path(zip_path); fingerprint=digest(zip_path); root=Path(base)/fingerprint[:16]; root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        files=[info for info in archive.infolist() if not info.is_dir()]
        for info in files:
            target=(root/info.filename).resolve()
            if target!=root.resolve() and root.resolve() not in target.parents: raise RuntimeError(f"Unsafe zip: {info.filename}")
        if any(not (root/info.filename).is_file() or (root/info.filename).stat().st_size!=info.file_size for info in files): archive.extractall(root)
        bad=[info.filename for info in files if not (root/info.filename).is_file() or (root/info.filename).stat().st_size!=info.file_size]
    if bad: raise RuntimeError(f"GT extraction incomplete: {bad[:5]}")
    meta={"zip":str(zip_path.resolve()),"zip_hash":fingerprint,"members":len(files),"image_members":sum(Path(info.filename).suffix.lower() in IMAGE_EXTENSIONS for info in files)}
    (root/"extraction.json").write_text(json.dumps(meta,indent=2)); return root,meta

def content_audit(frames, gt, out):
    rows=[]
    for source in ["closed","open"]:
        for row in frames[source].itertuples():
            path=Path(row.path).resolve()
            if not path.is_file(): raise RuntimeError(f"manifest image missing: {path}")
            rows.append({"source":source,"split":str(row.split),"label":str(row.label),"path":str(path),"sha256":digest(path)})
    for label,paths in gt.items():
        for path in paths: rows.append({"source":"gt","split":"gt","label":str(label),"path":str(path),"sha256":digest(path)})
    conflicts=[]
    for sha,group in pd.DataFrame(rows).groupby("sha256"):
        query={(row.source,row.split,row.label) for row in group.itertuples() if row.source!="gt"}; gt_labels={row.label for row in group.itertuples() if row.source=="gt"}
        if len(group)>1: conflicts.append({"sha256":sha,"query_roles":sorted(query),"gt_labels":sorted(gt_labels),"paths":sorted(group.path.unique())})
    pd.DataFrame(rows).to_csv(out/"content_audit.csv",index=False); (out/"content_conflicts.json").write_text(json.dumps(conflicts,indent=2))
    if conflicts: raise RuntimeError(f"image-content leakage detected; see {out/'content_conflicts.json'}")
    return {"images":len(rows),"unique_hashes":len({row['sha256'] for row in rows}),"conflicts":0}

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
                "manifest":str(manifest), "manifest_hash":digest(manifest) if manifest.is_file() else "", "stored_manifest_hash":args.get("manifest_sha256") or args.get("manifest_hash") or "", "manifest_labels":manifest_labels, "manifest_splits":manifest_splits, "manifest_error":manifest_error, "epoch":ck.get("epoch",0),
                "history":str(history),"history_epochs":history_epochs,"history_fields":history_fields,"score":(ck.get("metrics",{}) or {}).get("macro_f1",-1), "readable":True}
    except Exception as e: return {"checkpoint":str(path), "readable":False, "error":repr(e)}

def candidate_facts(item, target, target_hash, target_path):
    checkpoint, manifest = set(item.get("labels",[])), set(item.get("manifest_labels",[]))
    stored=item.get("stored_manifest_hash"); current=checkpoint==manifest==target and item.get("manifest_hash")==target_hash and Path(item.get("manifest","")).resolve()==Path(target_path).resolve(); historical=bool(stored) and stored==target_hash
    return current and (not stored or historical),historical,"training-time manifest hash exact" if historical else "current manifest exact; training-time hash unavailable"

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
    counts={"/".join(map(str,k)):int(v) for k,v in df.groupby(["partition","split","label"]).size().items()}; meta={"representation":representation,"total_features":total,"sampled":len(df),"max_samples":limit,"seed":seed,"method":"deterministic round-robin by partition/split/label","counts":counts,"csv":str(csv_path),"npz":str(npz_path),"sampling":str(sampling_path),"train_test_png":str(train_test_png),"valid_png":str(valid_png)}; sampling_path.write_text(json.dumps(meta,indent=2))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    label_names=sorted(df.label.unique()); colors={label:plt.cm.turbo(i/max(1,len(label_names)-1)) for i,label in enumerate(label_names)}
    def panels(splits, path):
        fig,axes=plt.subplots(2,len(splits),figsize=(7*len(splits),11),squeeze=False)
        for row,part in enumerate(["closed","open"]):
            for col,split in enumerate(splits):
                ax=axes[row,col]; q=df[(df.partition==part)&(df.split==split)]
                for label,g in q.groupby("label",sort=True): ax.scatter(g.x,g.y,s=9,alpha=.65,color=colors[label])
                ax.set_title(f"{name} [{representation.replace('_',' ')}]: {part} / {split}"); ax.set_xticks([]); ax.set_yticks([])
        handles=[Line2D([0],[0],marker="o",linestyle="",markersize=5,color=colors[label],label=label) for label in label_names]
        fig.legend(handles=handles,loc="center left",bbox_to_anchor=(.99,.5),fontsize=7); fig.tight_layout(rect=(0,0,.86,1)); fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)
    panels(["train","test"],train_test_png); panels(["valid"],valid_png)
    return meta

def axis_limits(values, bounded):
    low,high=float(values.min()),float(values.max()); span=high-low or max(abs(high)*.05,.02); pad=max(span*.12,.005 if bounded else span*.05)
    return (max(0.,low-pad),min(1.,high+pad)) if bounded else (low-pad,high+pad)

def save_curve(name, item, out):
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
    missing=sorted({"test_accuracy","test_macro_f1"}-set(df.columns)); status="partial" if missing else "ok"
    reason=f"history lacks per-epoch {', '.join(missing)}; available curves only" if missing else ""
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

def resume_tsne(out, seed, limit):
    progress_path=out/"progress.json"; progress=json.loads(progress_path.read_text()); failures=[]
    for name in MODELS:
        folder=out/"features"/name.replace(" ","_"); common={}; deployed={}
        try:
            for part in ["closed","open"]:
                path=folder/f"{part}_backbone.npz" if (folder/f"{part}_backbone.npz").is_file() else folder/f"{part}.npz"
                with np.load(path,allow_pickle=True) as archive: common[part]={key:archive[key] for key in archive.files}
            common_meta=save_tsne(name,common,out/"tsne",seed,limit,"common_backbone")
            if all((folder/f"{part}_backbone.npz").is_file() for part in ["closed","open"]):
                for part in ["closed","open"]:
                    with np.load(folder/f"{part}.npz",allow_pickle=True) as archive: deployed[part]={key:archive[key] for key in archive.files}
                deployed_meta={"status":"ok",**save_tsne(name,deployed,out/"tsne",seed,limit,"deployed")}
            else: deployed_meta={"status":"same_as_common_backbone",**common_meta}
            progress["models"].setdefault(name,{})["tsne"]={"status":"ok",**common_meta,"common_backbone":common_meta,"deployed":deployed_meta}
        except Exception as error:
            progress["models"].setdefault(name,{})["tsne"]={"status":"unavailable","reason":repr(error)}; failures.append(name)
        progress_path.write_text(json.dumps(progress,indent=2))
    if failures: raise RuntimeError(f"t-SNE incomplete for {failures}; saved features can be resumed")
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
    fingerprint=json.loads((out/"run_fingerprint.json").read_text())
    unverified=[name for name in MODELS if name!="ViT Base" and not selected.get(name,{}).get("own_manifest_backed")]
    tsne_note="all features" if args.tsne_max_samples<=0 else f"deterministic stratified sample, maximum {args.tsne_max_samples} per representation"
    def link(path,label):
        if not path: return ""
        path=Path(path)
        try: target=path.resolve().relative_to(out.resolve()).as_posix()
        except ValueError: target=str(path)
        return f"[{label}]({target})"
    lines=["# Metric Learning Diagnosis","",f"Artifacts: `{out.resolve()}`",f"Seed: {args.seed}; SS: {args.ss_per_label}; GT unmatched: 0; content leakage: 0",f"t-SNE: {tsne_note}; common-backbone plots cover all models, and deployed plots additionally cover representations that differ (Base preprocessing and SupCon projector). Full features remain in NPZ.","Preprocessing: ViT Base deployed results use its timm pretrained configuration; all common-backbone comparisons use the repository transform. Fine-tuned deployed results use that same training transform. Exact values are in progress.json.","SupCon deployed results use the model's L2-normalized projector output. Common-backbone results remove projector dimension/normalization confounding, but remain a training-configuration comparison because CE and SupCon runs also differ in batch size, LR, and augmentation.",f"Manifest provenance limitation: current manifest path/content is exact, but training-time manifest hashes are unavailable for {', '.join(unverified) if unverified else 'none'}.","","## Artifact index","- "+" · ".join([link(out/"summary.csv","deployed metrics"),link(out/"backbone_summary.csv","common-backbone metrics"),link(out/"selected_results.csv","validation-selected test results"),link(out/"predictions.csv","predictions")]),"- "+" · ".join([link(out/"curve_inventory.csv","curve inventory"),link(out/"prototype_membership.csv","prototype membership"),link(out/"checkpoint_selection.json","checkpoint selection"),link(out/"checkpoint_inventory.csv","checkpoint inventory")]),"- "+" · ".join([link(out/"run_fingerprint.json","full SHA-256 fingerprint"),link(out/"manifest_audit.json","manifest audit"),link(out/"provenance_audit.json","provenance audit"),link(out/"content_audit.csv","content hashes")]),"","## SHA-256 provenance","| Group | Item | SHA-256 |","|---|---|---|"]
    hash_groups=[("source",fingerprint.get("sources",{})),("checkpoint",fingerprint.get("checkpoints",{})),("manifest",fingerprint.get("manifests",{}))]
    if fingerprint.get("base",{}).get("sha256"): hash_groups.append(("base",{"ViT Base":fingerprint["base"]}))
    hash_groups.append(("GT",{"AI Hub zip":fingerprint.get("gt_zip",{})}))
    for group,items in hash_groups:
        for item,facts in items.items(): lines.append(f"| {group} | {item} | `{facts.get('sha256','unavailable')}` |")
    lines += ["","## Validation-selected joint setting, held-out test"]
    for row in chosen.itertuples(): lines.append(f"- {row.model} / {row.partition}: {row.prototype} + {row.distance}; validation accuracy {row.valid_accuracy:.4f}, macro-F1 {row.valid_macro_f1:.4f}; test accuracy {row.test_accuracy:.4f}, macro-F1 {row.test_macro_f1:.4f}")
    lines += ["","## Held-out test performance by partition","Values below are marginal means across all six prototype × distance settings."]
    for partition in ["closed","open"]:
        lines += ["",f"### {partition.title()}"]
        for name in MODELS:
            q=test_ok[(test_ok.model==name)&(test_ok.partition==partition)]; lines.append(f"- {name}: "+("unavailable" if q.empty else f"accuracy {q.accuracy.mean():.4f}, macro-F1 {q.macro_f1.mean():.4f}"))
    lines += ["","## Matched held-out test comparisons","Deltas are marginal means across matched prototype × distance settings."]
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
        assets=" · ".join(x for x in [link(curve.get("png"),"curve"),link(curve.get("csv"),"history")] if x)
        lines.append(f"- {curve['model']}: {details}"+(f"; {assets}" if assets else "")+(f" ({curve['reason']})" if curve.get("reason") else ""))
    lines += ["","## t-SNE coverage"]
    for name in MODELS:
        info=progress["models"].get(name,{}).get("tsne",{}); common=info.get("common_backbone",{}); deployed=info.get("deployed",{})
        assets=[link(common.get("train_test_png"),"common train/test"),link(common.get("valid_png"),"common valid")]
        if deployed.get("status")!="same_as_common_backbone": assets += [link(deployed.get("train_test_png"),"deployed train/test"),link(deployed.get("valid_png"),"deployed valid")]
        lines.append(f"- {name}: {info.get('status','unavailable')}"+("; "+" · ".join(x for x in assets if x) if any(assets) else ""))
    lines += ["",f"Completed deployed result rows: {len(ok)}/120",f"Completed common-backbone rows: {len(backbone_ok)}/120"]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n"); print(f"RESULT_ROWS={len(result)} OK_ROWS={len(ok)} BACKBONE_OK_ROWS={len(backbone_ok)}")
    return len(result)==len(ok)==len(backbone_result)==len(backbone_ok)==120,all(progress["models"].get(name,{}).get("tsne",{}).get("status")=="ok" for name in MODELS),len(ok),len(backbone_ok)

def write_completion(out):
    required=["REPORT.md","summary.csv","backbone_summary.csv","selected_results.csv","predictions.csv","prototype_membership.csv","curve_inventory.csv","checkpoint_selection.json","checkpoint_inventory.csv","run_fingerprint.json","manifest_audit.json","provenance_audit.json","content_audit.csv","content_conflicts.json","gt_mapping.csv","gt_manifest.csv","progress.json"]
    artifacts=set(out/name for name in required); curves=pd.read_csv(out/"curve_inventory.csv").fillna(""); progress=json.loads((out/"progress.json").read_text())
    selected_results=pd.read_csv(out/"selected_results.csv")
    if len(selected_results)!=10: raise RuntimeError(f"expected 10 validation-selected results, got {len(selected_results)}")
    for name in MODELS[1:]:
        rows=curves[curves.model==name]
        if len(rows)!=1 or rows.iloc[0].status not in {"ok","partial"}: raise RuntimeError(f"curve artifacts incomplete for {name}")
        artifacts.update([Path(rows.iloc[0].csv),Path(rows.iloc[0].png)])
    for name in MODELS:
        info=progress["models"].get(name,{}).get("tsne",{})
        if info.get("status")!="ok": raise RuntimeError(f"t-SNE incomplete for {name}")
        for representation in ["common_backbone","deployed"]:
            details=info.get(representation,{})
            if representation=="deployed" and details.get("status")=="same_as_common_backbone": continue
            for key in ["csv","npz","sampling","train_test_png","valid_png"]: artifacts.add(Path(details.get(key,"")))
    root=out.resolve(); recorded={}
    for path in artifacts:
        path=path.resolve()
        if not path.is_file() or (path!=root and root not in path.parents): raise RuntimeError(f"required report artifact missing or outside output: {path}")
        recorded[str(path.relative_to(root))]=digest(path)
    completion={"schema":1,"result_rows":120,"backbone_rows":120,"selected_rows":10,"fingerprint_sha256":digest(out/"run_fingerprint.json"),"artifacts":dict(sorted(recorded.items()))}
    temporary=out/".completion.json.tmp"; temporary.write_text(json.dumps(completion,indent=2)); temporary.replace(out/"completion.json")
    return completion

def main():
    p=argparse.ArgumentParser(); p.add_argument("--closed-manifest",required=True); p.add_argument("--open-manifest",required=True); p.add_argument("--full-manifest",required=True)
    p.add_argument("--checkpoint-root",action="append",required=True); p.add_argument("--base-checkpoint"); p.add_argument("--base-model-name",default="vit_large_patch16_224.augreg_in21k_ft_in1k")
    p.add_argument("--closed-checkpoint"); p.add_argument("--closed-supcon-checkpoint"); p.add_argument("--full-checkpoint"); p.add_argument("--full-supcon-checkpoint")
    p.add_argument("--gt-zip"); p.add_argument("--gt-root"); p.add_argument("--output-dir",required=True)
    p.add_argument("--ss-per-label",type=int,default=5); p.add_argument("--seed",type=int,default=42); p.add_argument("--batch-size",type=int,default=64); p.add_argument("--num-workers",type=int,default=8); p.add_argument("--tsne-max-samples",type=int,default=0); p.add_argument("--allow-cpu",action="store_true"); p.add_argument("--inventory-only",action="store_true"); p.add_argument("--resume-tsne",action="store_true"); a=p.parse_args()
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); (out/"completion.json").unlink(missing_ok=True); (out/"features").mkdir(exist_ok=True); (out/"tsne").mkdir(exist_ok=True); (out/"curves").mkdir(exist_ok=True)
    if a.resume_tsne:
        resume_tsne(out,a.seed,a.tsne_max_samples); numeric_ok,visual_ok,ok_rows,backbone_rows=write_report(out,a)
        if not numeric_ok or not visual_ok: raise RuntimeError(f"experiment incomplete after t-SNE resume: deployed {ok_rows}/120, backbone {backbone_rows}/120")
        write_completion(out)
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
    audit={"ok":not errors,"errors":errors,"split_errors":split_errors,"row_union_exact":row_union_exact,"manifests":{k:{"path":str(v.resolve()),"hash":digest(v),"rows":len(frames[k]),"labels":len(labels[k]),"splits":frames[k].split.value_counts().to_dict() if "split" in frames[k] else {}} for k,v in manifests.items()}}; (out/"manifest_audit.json").write_text(json.dumps(audit,indent=2))
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
            eligible,own,basis=candidate_facts(r,labels[part],audit["manifests"][part]["hash"],manifests[part])
            complete=name=="ViT-FullFT" or (r.get("history_epochs")==list(range(1,11)) and {"test_accuracy","test_macro_f1"}<=set(r.get("history_fields",[])))
            if r.get("readable") and r.get("kind")==kind and r.get("model_name")==a.base_model_name and r.get("image_size")==224 and (kind!="supcon" or r.get("loss_name") in ["","supcon"]) and eligible and complete: candidates.append((r,own,basis))
        if explicit[name]:
            wanted=str(Path(explicit[name]).resolve()); matches=[x for x in candidates if str(Path(x[0]["checkpoint"]).resolve())==wanted]
            if matches: candidates=matches
            else: selection_errors[name]=f"explicit checkpoint failed exact {part} manifest/completion validation: {wanted}"; candidates=[]
        if candidates:
            chosen,own,basis=max(candidates,key=lambda x:(x[1],set(["train","valid","test"])<=set(x[0]["manifest_splits"]),x[0]["epoch"],x[0]["score"]))
            selected[name]={**chosen,"selection_basis":("explicit " if explicit[name] else "")+basis,"missing_target_labels":[],"own_manifest_backed":own,"own_manifest_complete_splits":set(["train","valid","test"])<=set(chosen["manifest_splits"])}
    selected["ViT Base"]={"kind":"base","checkpoint":str(Path(a.base_checkpoint).resolve()),"model_name":a.base_model_name,"selection_basis":"explicit base checkpoint"} if a.base_checkpoint else {"kind":"pretrained_base","model_name":a.base_model_name,"image_size":224,"selection_basis":"timm pretrained weights"}
    (out/"checkpoint_selection.json").write_text(json.dumps(selected,indent=2)); (out/"checkpoint_selection_errors.json").write_text(json.dumps(selection_errors,indent=2)); print("SELECTED=",",".join(x for x in MODELS if x in selected)); print("MISSING=",",".join(x for x in MODELS if x not in selected))
    if a.inventory_only: return
    if not a.gt_zip or not a.gt_root: raise RuntimeError("GT paths required")
    if a.ss_per_label < 1: raise RuntimeError("ss-per-label must be positive")
    fingerprint_config={"base_model_name":a.base_model_name,"batch_size":a.batch_size,"num_workers":a.num_workers,"seed":a.seed,"ss_per_label":a.ss_per_label,"tsne_max_samples":a.tsne_max_samples,"representations":"deployed_and_common_backbone_v1"}
    fingerprint=run_fingerprint({name:selected[name]["checkpoint"] for name in needs if name in selected},manifests,a.gt_zip,fingerprint_config,a.base_checkpoint); (out/"run_fingerprint.json").write_text(json.dumps(fingerprint,indent=2))
    gtroot,gt_extract=extract_gt_zip(a.gt_zip,a.gt_root)
    gt,_=map_gt_label_folders(gtroot,labels["full"],out/"gt_mapping.csv"); gtmanifest=out/"gt_manifest.csv"; pd.DataFrame([{"path":str(p),"label":l,"split":"gt"} for l,ps in gt.items() for p in ps]).to_csv(gtmanifest,index=False)
    if set(gt)!=labels["full"]: raise RuntimeError("GT prototype coverage differs from full labels")
    leakage=content_audit(frames,gt,out); (out/"provenance_audit.json").write_text(json.dumps({"manifests":audit,"gt_extraction":gt_extract,"content":leakage},indent=2))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type=="cpu" and not a.allow_cpu: raise RuntimeError("CUDA unavailable")
    summary,backbone_summary,predictions,members,curves=[],[],[],[],[]; progress={"expected_result_rows":120,"expected_backbone_rows":120,"tsne_max_samples":a.tsne_max_samples,"models":{}}
    for name in MODELS:
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
            progress["models"][name]={"status":"ok","result_rows":24,"backbone_rows":24,"representation":representation,"preprocessing":preprocess,"tsne":{"status":"pending","representation":"common_backbone"}}
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
    try: progress=resume_tsne(out,a.seed,a.tsne_max_samples)
    except RuntimeError: progress=json.loads((out/"progress.json").read_text())
    numeric_ok,visual_ok,ok_rows,backbone_rows=write_report(out,a)
    if not numeric_ok or not visual_ok: raise RuntimeError(f"experiment incomplete: deployed {ok_rows}/120, backbone {backbone_rows}/120, t-SNE complete={visual_ok}; partial artifacts preserved in {out}")
    write_completion(out)

if __name__ == "__main__": main()

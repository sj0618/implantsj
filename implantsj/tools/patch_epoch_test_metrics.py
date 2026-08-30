#!/usr/bin/env python3
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PATCHES = {
    "src/supervised.py": [
        (
            "    valid_loader = create_loader(args.manifest, 'valid', class_to_idx, args.image_size, args.batch_size, args.num_workers, train=False, strong_aug=False, return_path=True)\n\n"
            "    model = create_classifier(\n",
            "    valid_loader = create_loader(args.manifest, 'valid', class_to_idx, args.image_size, args.batch_size, args.num_workers, train=False, strong_aug=False, return_path=True)\n"
            "    test_loader = create_loader(args.manifest, 'test', class_to_idx, args.image_size, args.batch_size, args.num_workers, train=False, strong_aug=False, return_path=True)\n\n"
            "    model = create_classifier(\n",
        ),
        (
            "        valid_metrics = evaluate_classifier(model, valid_loader, device, idx_to_class, 'valid', out_dir=None)\n"
            "        score = float(valid_metrics.get(args.monitor, valid_metrics.get('macro_f1', 0.0)))\n",
            "        valid_metrics = evaluate_classifier(model, valid_loader, device, idx_to_class, 'valid', out_dir=None)\n"
            "        test_metrics = evaluate_classifier(model, test_loader, device, idx_to_class, 'test', out_dir=None)\n"
            "        score = float(valid_metrics.get(args.monitor, valid_metrics.get('macro_f1', 0.0)))\n",
        ),
        (
            "            'valid_weighted_f1': valid_metrics['weighted_f1'],\n"
            "            'valid_macro_precision': valid_metrics['macro_precision'],\n",
            "            'valid_weighted_f1': valid_metrics['weighted_f1'],\n"
            "            'test_accuracy': test_metrics['accuracy'],\n"
            "            'test_macro_f1': test_metrics['macro_f1'],\n"
            "            'test_weighted_f1': test_metrics['weighted_f1'],\n"
            "            'valid_macro_precision': valid_metrics['macro_precision'],\n",
        ),
    ],
    "src/metric_learning.py": [
        (
            "from .data import ImageClassificationDataset, build_transforms, label_mapping, load_manifest\n",
            "from .data import ImageClassificationDataset, build_transforms, create_loader, label_mapping, load_manifest\n",
        ),
        (
            "def evaluate_metric_model(model, manifest: str, class_to_idx: Dict[str, int], image_size: int, batch_size: int, num_workers: int, device, out_dir: Optional[Path] = None, split: str = 'valid'):\n"
            "    from .data import create_loader\n"
            "    idx_to_class = {i: c for c, i in class_to_idx.items()}\n"
            "    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]\n"
            "    train_loader = create_loader(manifest, 'train', class_to_idx, image_size, batch_size, num_workers, train=False, return_path=True)\n"
            "    query_loader = create_loader(manifest, split, class_to_idx, image_size, batch_size, num_workers, train=False, return_path=True)\n"
            "    train_z, train_y, _ = extract_embeddings(model, train_loader, device)\n"
            "    query_z, query_y, paths = extract_embeddings(model, query_loader, device)\n",
            "def evaluate_metric_model(model, manifest: str, class_to_idx: Dict[str, int], image_size: int, batch_size: int, num_workers: int, device, out_dir: Optional[Path] = None, split: str = 'valid', support=None):\n"
            "    idx_to_class = {i: c for c, i in class_to_idx.items()}\n"
            "    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]\n"
            "    query_loader = create_loader(manifest, split, class_to_idx, image_size, batch_size, num_workers, train=False, return_path=True)\n"
            "    if support is None:\n"
            "        train_loader = create_loader(manifest, 'train', class_to_idx, image_size, batch_size, num_workers, train=False, return_path=True)\n"
            "        train_z, train_y, _ = extract_embeddings(model, train_loader, device)\n"
            "    else:\n"
            "        train_z, train_y = support\n"
            "    query_z, query_y, paths = extract_embeddings(model, query_loader, device)\n",
        ),
        (
            "    ds = TwoViewDataset(args.manifest, 'train', class_to_idx, args.image_size)\n"
            "    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), drop_last=True)\n"
            "    model = MetricModel(args.model_name, embedding_dim=args.embedding_dim, pretrained=args.pretrained, dropout=args.dropout, drop_path_rate=args.drop_path_rate).to(device)\n",
            "    ds = TwoViewDataset(args.manifest, 'train', class_to_idx, args.image_size)\n"
            "    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), drop_last=True)\n"
            "    eval_train_loader = create_loader(args.manifest, 'train', class_to_idx, args.image_size, args.batch_size, args.num_workers, train=False, return_path=True)\n"
            "    model = MetricModel(args.model_name, embedding_dim=args.embedding_dim, pretrained=args.pretrained, dropout=args.dropout, drop_path_rate=args.drop_path_rate).to(device)\n",
        ),
        (
            "        valid_metrics = evaluate_metric_model(model, args.manifest, class_to_idx, args.image_size, args.batch_size, args.num_workers, device, out_dir=None, split='valid')\n"
            "        score = valid_metrics['macro_f1']\n",
            "        support = extract_embeddings(model, eval_train_loader, device)[:2]\n"
            "        valid_metrics = evaluate_metric_model(model, args.manifest, class_to_idx, args.image_size, args.batch_size, args.num_workers, device, out_dir=None, split='valid', support=support)\n"
            "        test_metrics = evaluate_metric_model(model, args.manifest, class_to_idx, args.image_size, args.batch_size, args.num_workers, device, out_dir=None, split='test', support=support)\n"
            "        score = valid_metrics['macro_f1']\n",
        ),
        (
            "        row = {'epoch': epoch, 'train_loss': train_loss, 'valid_accuracy': valid_metrics['accuracy'], 'valid_macro_f1': valid_metrics['macro_f1'], 'lr': optimizer.param_groups[0]['lr'], 'seconds': time.time()-t0}\n",
            "        row = {\n"
            "            'epoch': epoch, 'train_loss': train_loss,\n"
            "            'valid_accuracy': valid_metrics['accuracy'], 'valid_macro_f1': valid_metrics['macro_f1'], 'valid_weighted_f1': valid_metrics['weighted_f1'],\n"
            "            'test_accuracy': test_metrics['accuracy'], 'test_macro_f1': test_metrics['macro_f1'], 'test_weighted_f1': test_metrics['weighted_f1'],\n"
            "            'lr': optimizer.param_groups[0]['lr'], 'seconds': time.time()-t0,\n"
            "        }\n",
        ),
        (
            "    best = torch.load(run_dir / 'checkpoints' / 'best.pt', map_location='cpu')\n"
            "    model.load_state_dict(best['model_state_dict'], strict=True)\n"
            "    evaluate_metric_model(model, args.manifest, class_to_idx, args.image_size, args.batch_size, args.num_workers, device, out_dir=run_dir, split='valid')\n"
            "    try:\n"
            "        test_metrics = evaluate_metric_model(model, args.manifest, class_to_idx, args.image_size, args.batch_size, args.num_workers, device, out_dir=run_dir, split='test')\n",
            "    best = torch.load(run_dir / 'checkpoints' / 'best.pt', map_location='cpu')\n"
            "    model.load_state_dict(best['model_state_dict'], strict=True)\n"
            "    support = extract_embeddings(model, eval_train_loader, device)[:2]\n"
            "    evaluate_metric_model(model, args.manifest, class_to_idx, args.image_size, args.batch_size, args.num_workers, device, out_dir=run_dir, split='valid', support=support)\n"
            "    try:\n"
            "        test_metrics = evaluate_metric_model(model, args.manifest, class_to_idx, args.image_size, args.batch_size, args.num_workers, device, out_dir=run_dir, split='test', support=support)\n",
        ),
    ],
}


def prepare(path, replacements):
    original = path.read_text(encoding="utf-8")
    result = original
    states = []
    for old, new in replacements:
        if new in result:
            states.append("patched")
        elif old in result:
            states.append("original")
            result = result.replace(old, new, 1)
        else:
            raise RuntimeError(f"unexpected source: {path}")
    if len(set(states)) != 1:
        raise RuntimeError(f"partially patched source refused: {path}")
    compile(result, str(path), "exec")
    return original, result, states[0], path.stat().st_mode


def main():
    plans = []
    for relative, replacements in PATCHES.items():
        path = ROOT / relative
        plans.append((path, *prepare(path, replacements)))
    states = {plan[3] for plan in plans}
    if len(states) != 1:
        raise RuntimeError("partially patched repository refused")
    if states == {"patched"}:
        print("epoch test metrics already patched")
        return

    staged = []
    replaced = []
    try:
        for path, original, result, _, mode in plans:
            temp = path.with_name(path.name + ".metricdiag.tmp")
            temp.write_text(result, encoding="utf-8")
            os.chmod(temp, mode)
            staged.append(temp)
        for temp, plan in zip(staged, plans):
            path, original, _, _, mode = plan
            os.replace(temp, path)
            replaced.append((path, original, mode))
    except Exception:
        for path, original, mode in replaced:
            path.write_text(original, encoding="utf-8")
            os.chmod(path, mode)
        raise
    finally:
        for temp in staged:
            temp.unlink(missing_ok=True)
    print("patched per-epoch test metrics in supervised and metric training")


if __name__ == "__main__":
    main()

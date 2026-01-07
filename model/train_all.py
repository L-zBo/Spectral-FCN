# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from sklearn.metrics import f1_score, recall_score, precision_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

from FCN import FaultClassificationNetwork, FaultMultiHeadNetwork


def read_spectrum_csv(path: str) -> np.ndarray:
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try:
            df = pd.read_csv(path, encoding=enc, header=None, engine="python", sep=",")
            s = pd.to_numeric(df.iloc[:, -1], errors="coerce").dropna()
            arr = s.to_numpy(dtype=np.float32)
            if arr.size > 0:
                return arr
        except Exception as e:
            continue
    raise RuntimeError(f"无法读取CSV: {path}")


def standardize(x: np.ndarray, method: str = "zscore") -> np.ndarray:
    if method == "zscore":
        m, s = np.mean(x), np.std(x)
        return ((x - m) / (s + 1e-12)).astype(np.float32)
    elif method == "minmax":
        xmin, xmax = np.min(x), np.max(x)
        return ((x - xmin) / (xmax - xmin + 1e-12)).astype(np.float32)
    return x.astype(np.float32)


class SpectrumDataset(Dataset):
    def __init__(self, files: List[str], labels: List[int], standardize_method: str = "zscore"):
        self.files = files
        self.labels = labels
        self.standardize_method = standardize_method

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = read_spectrum_csv(self.files[idx])
        x = standardize(x, self.standardize_method)
        return torch.from_numpy(x), self.labels[idx]


class DualLabelDataset(Dataset):
    """双标签数据集，用于 PP+PE 双任务学习"""
    def __init__(self, files: List[str], pp_labels: List[int], pe_labels: List[int],
                 standardize_method: str = "zscore"):
        self.files = files
        self.pp_labels = pp_labels
        self.pe_labels = pe_labels
        self.standardize_method = standardize_method

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = read_spectrum_csv(self.files[idx])
        x = standardize(x, self.standardize_method)
        return torch.from_numpy(x), self.pp_labels[idx], self.pe_labels[idx]


def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    B = len(batch)
    X = torch.zeros(B, 1, max_len, dtype=torch.float32)
    Y = torch.tensor(ys, dtype=torch.long)
    for i, x in enumerate(xs):
        X[i, 0, :x.shape[0]] = x
    return X, Y


def dual_label_collate_fn(batch):
    """双标签数据的 collate 函数"""
    xs, pp_ys, pe_ys = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    B = len(batch)
    X = torch.zeros(B, 1, max_len, dtype=torch.float32)
    Y_pp = torch.tensor(pp_ys, dtype=torch.long)
    Y_pe = torch.tensor(pe_ys, dtype=torch.long)
    for i, x in enumerate(xs):
        X[i, 0, :x.shape[0]] = x
    return X, Y_pp, Y_pe


def collect_csv_files(directory: str) -> List[str]:
    files = []
    for root, _, filenames in os.walk(directory):
        for f in filenames:
            if f.lower().endswith('.csv'):
                files.append(os.path.join(root, f))
    return sorted(files)


def split_files(files: List[str], train_n: int, seed: int) -> Tuple[List[str], List[str]]:
    rnd = random.Random(seed)
    f = list(files)
    rnd.shuffle(f)
    return f[:train_n], f[train_n:]


def get_three_class_loaders(root: str, mix_type: str, batch_size: int,
                            standardize_method: str, seed: int, num_workers: int):
    base_dir = os.path.join(root, f"{mix_type}+淀粉")

    pure_dir = os.path.join(base_dir, "无污染")
    slight_dir = os.path.join(base_dir, "轻微浓度")
    severe_dir = os.path.join(base_dir, "严重浓度")

    pure_files = collect_csv_files(pure_dir)
    slight_files = collect_csv_files(slight_dir)
    severe_files = collect_csv_files(severe_dir)

    print(f"  无污染: {len(pure_files)} 文件")
    print(f"  轻微浓度: {len(slight_files)} 文件")
    print(f"  严重浓度: {len(severe_files)} 文件")

    required, train_n = 30, 22
    pure_tr, pure_va = split_files(pure_files[:required], train_n, seed)
    slight_tr, slight_va = split_files(slight_files[:required], train_n, seed)
    severe_tr, severe_va = split_files(severe_files[:required], train_n, seed)

    train_files = pure_tr + slight_tr + severe_tr
    train_labels = [0]*len(pure_tr) + [1]*len(slight_tr) + [2]*len(severe_tr)
    val_files = pure_va + slight_va + severe_va
    val_labels = [0]*len(pure_va) + [1]*len(slight_va) + [2]*len(severe_va)

    train_ds = SpectrumDataset(train_files, train_labels, standardize_method)
    val_ds = SpectrumDataset(val_files, val_labels, standardize_method)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn,
                            pin_memory=torch.cuda.is_available())

    meta = {
        "class_names": {0: "无污染", 1: "轻微浓度", 2: "严重浓度"},
        "train_counts": {0: len(pure_tr), 1: len(slight_tr), 2: len(severe_tr)},
        "val_counts": {0: len(pure_va), 1: len(slight_va), 2: len(severe_va)},
    }

    return train_loader, val_loader, meta


def get_two_class_loaders(root: str, sub_type: str, batch_size: int,
                          standardize_method: str, seed: int, num_workers: int):
    base_dir = os.path.join(root, "PP+PE+淀粉")

    slight_keyword = f"轻微{sub_type}"
    severe_keyword = f"严重{sub_type}"

    slight_files = []
    severe_files = []

    for d in os.listdir(base_dir):
        full_path = os.path.join(base_dir, d)
        if os.path.isdir(full_path):
            if slight_keyword in d:
                slight_files.extend(collect_csv_files(full_path))
            elif severe_keyword in d:
                severe_files.extend(collect_csv_files(full_path))

    print(f"  {slight_keyword}: {len(slight_files)} 文件")
    print(f"  {severe_keyword}: {len(severe_files)} 文件")

    required, train_n = 50, 40
    slight_tr, slight_va = split_files(slight_files[:required], train_n, seed)
    severe_tr, severe_va = split_files(severe_files[:required], train_n, seed)

    train_files = slight_tr + severe_tr
    train_labels = [0]*len(slight_tr) + [1]*len(severe_tr)
    val_files = slight_va + severe_va
    val_labels = [0]*len(slight_va) + [1]*len(severe_va)

    train_ds = SpectrumDataset(train_files, train_labels, standardize_method)
    val_ds = SpectrumDataset(val_files, val_labels, standardize_method)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn,
                            pin_memory=torch.cuda.is_available())

    meta = {
        "class_names": {0: f"{sub_type}轻微浓度", 1: f"{sub_type}严重浓度"},
        "train_counts": {0: len(slight_tr), 1: len(severe_tr)},
        "val_counts": {0: len(slight_va), 1: len(severe_va)},
    }

    return train_loader, val_loader, meta


def get_pp_pe_dual_task_loaders(root: str, batch_size: int,
                                 standardize_method: str, seed: int, num_workers: int):
    """
    双任务数据加载器：同时输出 PP 和 PE 的浓度标签

    标签定义:
    - PP: 0=无污染, 1=轻微, 2=严重
    - PE: 0=无污染, 1=轻微, 2=严重

    输出示例: (1, 2) 表示 PP:轻微, PE:严重
    """
    base_dir = os.path.join(root, "PP+PE+淀粉")

    # 定义文件夹与双标签的映射
    # 格式: (文件夹名, PP标签, PE标签)
    folder_label_map = [
        ("无污染", 0, 0),                    # PP:无, PE:无
        ("轻微PP+轻微PE+淀粉", 1, 1),         # PP:轻微, PE:轻微
        ("严重PP+严重PE+淀粉", 2, 2),         # PP:严重, PE:严重
        # 以下为预留的交叉组合（当数据可用时取消注释）
        # ("轻微PP+严重PE+淀粉", 1, 2),       # PP:轻微, PE:严重
        # ("严重PP+轻微PE+淀粉", 2, 1),       # PP:严重, PE:轻微
    ]

    all_files = []
    all_pp_labels = []
    all_pe_labels = []

    print("  双任务数据加载 (PP + PE):")
    for folder_name, pp_label, pe_label in folder_label_map:
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.exists(folder_path):
            files = collect_csv_files(folder_path)
            print(f"    {folder_name}: {len(files)} 文件 -> PP:{pp_label}, PE:{pe_label}")
            all_files.extend(files)
            all_pp_labels.extend([pp_label] * len(files))
            all_pe_labels.extend([pe_label] * len(files))
        else:
            print(f"    {folder_name}: 文件夹不存在，跳过")

    # 打乱并分割数据
    combined = list(zip(all_files, all_pp_labels, all_pe_labels))
    rnd = random.Random(seed)
    rnd.shuffle(combined)

    # 按比例分割 (约 75% 训练, 25% 验证)
    train_ratio = 0.75
    train_n = int(len(combined) * train_ratio)

    train_data = combined[:train_n]
    val_data = combined[train_n:]

    train_files, train_pp, train_pe = zip(*train_data) if train_data else ([], [], [])
    val_files, val_pp, val_pe = zip(*val_data) if val_data else ([], [], [])

    train_files, train_pp, train_pe = list(train_files), list(train_pp), list(train_pe)
    val_files, val_pp, val_pe = list(val_files), list(val_pp), list(val_pe)

    print(f"  训练集: {len(train_files)} 样本, 验证集: {len(val_files)} 样本")

    train_ds = DualLabelDataset(train_files, train_pp, train_pe, standardize_method)
    val_ds = DualLabelDataset(val_files, val_pp, val_pe, standardize_method)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=dual_label_collate_fn,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=dual_label_collate_fn,
                            pin_memory=torch.cuda.is_available())

    meta = {
        "task_type": "dual_task",
        "pp_class_names": {0: "无污染", 1: "轻微浓度", 2: "严重浓度"},
        "pe_class_names": {0: "无污染", 1: "轻微浓度", 2: "严重浓度"},
        "train_samples": len(train_files),
        "val_samples": len(val_files),
    }

    return train_loader, val_loader, meta


def get_pp_pe_three_class_loaders(root: str, batch_size: int,
                                   standardize_method: str, seed: int, num_workers: int):
    base_dir = os.path.join(root, "PP+PE+淀粉")

    pure_dir = os.path.join(base_dir, "无污染")
    slight_dir = os.path.join(base_dir, "轻微PP+轻微PE+淀粉")
    severe_dir = os.path.join(base_dir, "严重PP+严重PE+淀粉")

    pure_files = collect_csv_files(pure_dir)
    slight_files = collect_csv_files(slight_dir)
    severe_files = collect_csv_files(severe_dir)

    print(f"  无污染: {len(pure_files)} 文件")
    print(f"  轻微PP+轻微PE: {len(slight_files)} 文件")
    print(f"  严重PP+严重PE: {len(severe_files)} 文件")

    required, train_n = 30, 22
    pure_tr, pure_va = split_files(pure_files[:required], train_n, seed)
    slight_tr, slight_va = split_files(slight_files[:required], train_n, seed)
    severe_tr, severe_va = split_files(severe_files[:required], train_n, seed)

    train_files = pure_tr + slight_tr + severe_tr
    train_labels = [0]*len(pure_tr) + [1]*len(slight_tr) + [2]*len(severe_tr)
    val_files = pure_va + slight_va + severe_va
    val_labels = [0]*len(pure_va) + [1]*len(slight_va) + [2]*len(severe_va)

    train_ds = SpectrumDataset(train_files, train_labels, standardize_method)
    val_ds = SpectrumDataset(val_files, val_labels, standardize_method)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn,
                            pin_memory=torch.cuda.is_available())

    meta = {
        "class_names": {0: "无污染", 1: "轻微浓度", 2: "严重浓度"},
        "train_counts": {0: len(pure_tr), 1: len(slight_tr), 2: len(severe_tr)},
        "val_counts": {0: len(pure_va), 1: len(slight_va), 2: len(severe_va)},
    }

    return train_loader, val_loader, meta


def set_seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {"f1_macro": None, "recall_macro": None, "precision_macro": None}
    y_true_np = np.array(y_true, dtype=np.int64)
    y_pred_np = np.array(y_pred, dtype=np.int64)
    if SKLEARN_AVAILABLE:
        f1m = float(f1_score(y_true_np, y_pred_np, average="macro"))
        recall = float(recall_score(y_true_np, y_pred_np, average="macro"))
        precision = float(precision_score(y_true_np, y_pred_np, average="macro"))
    else:
        f1m, recall, precision = None, None, None
    return {"f1_macro": f1m, "recall_macro": recall, "precision_macro": precision}


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_samples, total_correct = 0.0, 0, 0
    for xb, yb in loader:
        xb = xb.to(device, dtype=torch.float32)
        yb = yb.to(device, dtype=torch.long)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        bs = yb.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        total_correct += (logits.argmax(1) == yb).sum().item()
    return {"loss": total_loss/total_samples, "acc": total_correct/total_samples}


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_samples, total_correct = 0.0, 0, 0
    all_true, all_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, dtype=torch.float32)
            yb = yb.to(device, dtype=torch.long)
            logits = model(xb)
            loss = criterion(logits, yb)
            bs = yb.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
            preds = logits.argmax(1)
            total_correct += (preds == yb).sum().item()
            all_true.extend(yb.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
    metrics = evaluate_metrics(all_true, all_pred)
    return {"loss": total_loss/total_samples, "acc": total_correct/total_samples, **metrics}


# ==================== 双任务训练函数 ====================

def train_dual_task_one_epoch(model, loader, criterion, optimizer, device):
    """双任务模型的单轮训练"""
    model.train()
    total_loss, total_samples = 0.0, 0
    pp_correct, pe_correct = 0, 0

    for xb, yb_pp, yb_pe in loader:
        xb = xb.to(device, dtype=torch.float32)
        yb_pp = yb_pp.to(device, dtype=torch.long)
        yb_pe = yb_pe.to(device, dtype=torch.long)

        pp_logits, pe_logits = model(xb)

        # 双任务损失：PP损失 + PE损失
        loss_pp = criterion(pp_logits, yb_pp)
        loss_pe = criterion(pe_logits, yb_pe)
        loss = loss_pp + loss_pe

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = yb_pp.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        pp_correct += (pp_logits.argmax(1) == yb_pp).sum().item()
        pe_correct += (pe_logits.argmax(1) == yb_pe).sum().item()

    return {
        "loss": total_loss / total_samples,
        "pp_acc": pp_correct / total_samples,
        "pe_acc": pe_correct / total_samples,
        "avg_acc": (pp_correct + pe_correct) / (2 * total_samples)
    }


def validate_dual_task_one_epoch(model, loader, criterion, device):
    """双任务模型的单轮验证"""
    model.eval()
    total_loss, total_samples = 0.0, 0
    pp_correct, pe_correct = 0, 0
    pp_true, pp_pred = [], []
    pe_true, pe_pred = [], []

    with torch.no_grad():
        for xb, yb_pp, yb_pe in loader:
            xb = xb.to(device, dtype=torch.float32)
            yb_pp = yb_pp.to(device, dtype=torch.long)
            yb_pe = yb_pe.to(device, dtype=torch.long)

            pp_logits, pe_logits = model(xb)

            loss_pp = criterion(pp_logits, yb_pp)
            loss_pe = criterion(pe_logits, yb_pe)
            loss = loss_pp + loss_pe

            bs = yb_pp.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            pp_preds = pp_logits.argmax(1)
            pe_preds = pe_logits.argmax(1)

            pp_correct += (pp_preds == yb_pp).sum().item()
            pe_correct += (pe_preds == yb_pe).sum().item()

            pp_true.extend(yb_pp.cpu().tolist())
            pp_pred.extend(pp_preds.cpu().tolist())
            pe_true.extend(yb_pe.cpu().tolist())
            pe_pred.extend(pe_preds.cpu().tolist())

    pp_metrics = evaluate_metrics(pp_true, pp_pred)
    pe_metrics = evaluate_metrics(pe_true, pe_pred)

    return {
        "loss": total_loss / total_samples,
        "pp_acc": pp_correct / total_samples,
        "pe_acc": pe_correct / total_samples,
        "avg_acc": (pp_correct + pe_correct) / (2 * total_samples),
        "pp_f1_macro": pp_metrics["f1_macro"],
        "pe_f1_macro": pe_metrics["f1_macro"],
        "pp_recall_macro": pp_metrics["recall_macro"],
        "pe_recall_macro": pe_metrics["recall_macro"],
    }


def train_dual_task_model(task_name, loader_fn, loader_args, output_dir, args):
    """双任务模型训练主函数"""
    print(f"\n{'='*60}")
    print(f"Training (Dual-Task): {task_name}")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    set_seed_all(args.seed)

    ckpt_dir = os.path.join(output_dir, "checkpoints", "best")
    os.makedirs(ckpt_dir, exist_ok=True)

    print("Loading data...")
    train_loader, val_loader, meta = loader_fn(**loader_args)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 使用双头网络：PP 3类, PE 3类
    model = FaultMultiHeadNetwork(num_classes_pp=3, num_classes_pe=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    history = []
    best = {"epoch": 0, "val_avg_acc": -1.0, "val_pp_acc": None, "val_pe_acc": None}
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = train_dual_task_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m = validate_dual_task_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        hist = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]['lr'],
            "train_loss": train_m["loss"],
            "train_pp_acc": train_m["pp_acc"],
            "train_pe_acc": train_m["pe_acc"],
            "train_avg_acc": train_m["avg_acc"],
            "val_loss": val_m["loss"],
            "val_pp_acc": val_m["pp_acc"],
            "val_pe_acc": val_m["pe_acc"],
            "val_avg_acc": val_m["avg_acc"],
            "pp_f1_macro": val_m["pp_f1_macro"],
            "pe_f1_macro": val_m["pe_f1_macro"],
            "time_sec": time.time() - t0
        }
        history.append(hist)

        pp_f1_str = f"{val_m['pp_f1_macro']:.4f}" if val_m['pp_f1_macro'] else "N/A"
        pe_f1_str = f"{val_m['pe_f1_macro']:.4f}" if val_m['pe_f1_macro'] else "N/A"
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train: loss={train_m['loss']:.4f} PP={train_m['pp_acc']:.4f} PE={train_m['pe_acc']:.4f} | "
              f"Val: loss={val_m['loss']:.4f} PP={val_m['pp_acc']:.4f} PE={val_m['pe_acc']:.4f} "
              f"F1(PP)={pp_f1_str} F1(PE)={pe_f1_str}")

        # 使用平均准确率作为最佳模型选择标准
        if val_m["avg_acc"] > best["val_avg_acc"]:
            best = {
                "epoch": epoch,
                "val_avg_acc": val_m["avg_acc"],
                "val_pp_acc": val_m["pp_acc"],
                "val_pe_acc": val_m["pe_acc"],
                "pp_f1_macro": val_m["pp_f1_macro"],
                "pe_f1_macro": val_m["pe_f1_macro"],
            }
            model.save_weights_multi(ckpt_dir)
            print(f"  -> New best! Saved to {ckpt_dir}")
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement")
            break

    summary = {
        "args": vars(args),
        "meta": meta,
        "epochs": history,
        "best": {
            "best_epoch": best["epoch"],
            "val_avg_acc": best["val_avg_acc"],
            "val_pp_acc": best["val_pp_acc"],
            "val_pe_acc": best["val_pe_acc"],
            "pp_f1_macro": best["pp_f1_macro"],
            "pe_f1_macro": best["pe_f1_macro"],
        },
        "total_time_sec": sum(h["time_sec"] for h in history)
    }

    with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nCompleted: {task_name}")
    print(f"Best: Epoch={best['epoch']}, Avg_Acc={best['val_avg_acc']:.4f}, "
          f"PP_Acc={best['val_pp_acc']:.4f}, PE_Acc={best['val_pe_acc']:.4f}")
    return best


def train_model(task_name, loader_fn, loader_args, num_classes, output_dir, args):
    print(f"\n{'='*60}")
    print(f"Training: {task_name}")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    set_seed_all(args.seed)

    ckpt_dir = os.path.join(output_dir, "checkpoints", "best")
    os.makedirs(ckpt_dir, exist_ok=True)

    print("Loading data...")
    train_loader, val_loader, meta = loader_fn(**loader_args)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    model = FaultClassificationNetwork(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    history = []
    best = {"epoch": 0, "val_acc": -1.0, "val_f1_macro": None}
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        hist = {"epoch": epoch, "lr": optimizer.param_groups[0]['lr'],
                "train_loss": train_m["loss"], "train_acc": train_m["acc"],
                "val_loss": val_m["loss"], "val_acc": val_m["acc"],
                "f1_macro": val_m["f1_macro"], "recall_macro": val_m["recall_macro"],
                "precision_macro": val_m["precision_macro"], "time_sec": time.time() - t0}
        history.append(hist)

        f1_str = f"{val_m['f1_macro']:.4f}" if val_m['f1_macro'] else "N/A"
        recall_str = f"{val_m['recall_macro']:.4f}" if val_m['recall_macro'] else "N/A"
        print(f"Epoch {epoch:3d}/{args.epochs} | Train: loss={train_m['loss']:.4f} acc={train_m['acc']:.4f} | "
              f"Val: loss={val_m['loss']:.4f} acc={val_m['acc']:.4f} f1={f1_str} recall={recall_str}")

        if val_m["acc"] > best["val_acc"]:
            best = {"epoch": epoch, "val_acc": val_m["acc"], "val_f1_macro": val_m["f1_macro"],
                    "val_recall_macro": val_m["recall_macro"], "val_precision_macro": val_m["precision_macro"]}
            model.save_weights(ckpt_dir)
            print(f"  -> New best! Saved to {ckpt_dir}")
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement")
            break

    summary = {"args": vars(args), "meta": meta, "epochs": history,
               "best": {"best_epoch": best["epoch"], "val_acc": best["val_acc"],
                        "f1_macro": best["val_f1_macro"], "recall_macro": best.get("val_recall_macro"),
                        "precision_macro": best.get("val_precision_macro")},
               "total_time_sec": sum(h["time_sec"] for h in history)}

    with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nCompleted: {task_name}")
    print(f"Best: Epoch={best['epoch']}, Acc={best['val_acc']:.4f}, F1={best['val_f1_macro']}")
    return best


def main():
    parser = argparse.ArgumentParser(description="Train all models")
    parser.add_argument("--root", type=str, default="dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--standardize", type=str, default="zscore")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--output-root", type=str, default="new_output")
    parser.add_argument("--task", type=str, default="all",
                        choices=["all", "pp", "pe", "pp_pe"])
    args = parser.parse_args()

    print("="*60)
    print("Raman Spectrum Classification Training")
    print("="*60)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Output: {args.output_root}")

    common_args = {"batch_size": args.batch_size, "standardize_method": args.standardize,
                   "seed": args.seed, "num_workers": args.num_workers}

    # 单任务配置 (PP, PE)
    single_tasks = {
        "pp": ("PP+Starch", get_three_class_loaders,
               {"root": args.root, "mix_type": "PP", **common_args}, 3, "PP_Starch"),
        "pe": ("PE+Starch", get_three_class_loaders,
               {"root": args.root, "mix_type": "PE", **common_args}, 3, "PE_Starch"),
    }

    # 双任务配置 (PP+PE)
    dual_task_config = {
        "pp_pe": ("PP+PE+Starch (Dual-Task)", get_pp_pe_dual_task_loaders,
                  {"root": args.root, **common_args}, "PP_PE_Starch"),
    }

    results = {}
    task_list = ["pp", "pe", "pp_pe"] if args.task == "all" else [args.task]

    for task_key in task_list:
        if task_key in single_tasks:
            # 单任务训练
            name, loader_fn, loader_args, num_classes, dir_name = single_tasks[task_key]
            output_dir = os.path.join(args.output_root, dir_name)
            best = train_model(name, loader_fn, loader_args, num_classes, output_dir, args)
            results[task_key] = {"type": "single", "best": best}
        elif task_key in dual_task_config:
            # 双任务训练
            name, loader_fn, loader_args, dir_name = dual_task_config[task_key]
            output_dir = os.path.join(args.output_root, dir_name)
            best = train_dual_task_model(name, loader_fn, loader_args, output_dir, args)
            results[task_key] = {"type": "dual", "best": best}

    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    for k, v in results.items():
        if v["type"] == "single":
            best = v["best"]
            name = single_tasks[k][0]
            print(f"{name}: Epoch={best['epoch']}, Acc={best['val_acc']:.4f}, F1={best['val_f1_macro']}")
        else:
            best = v["best"]
            name = dual_task_config[k][0]
            print(f"{name}: Epoch={best['epoch']}, "
                  f"PP_Acc={best['val_pp_acc']:.4f}, PE_Acc={best['val_pe_acc']:.4f}, "
                  f"Avg_Acc={best['val_avg_acc']:.4f}")
    print(f"\nAll results saved to: {args.output_root}")


if __name__ == "__main__":
    main()

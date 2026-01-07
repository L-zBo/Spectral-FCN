# -*- coding: utf-8 -*-
import os
import re
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

def read_spectrum_csv(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise RuntimeError(f"找不到文件: {path}")
    last_err: Optional[Exception] = None
    for enc in ("utf-8-sig", "gbk"):
        try:
            df = pd.read_csv(path, encoding=enc, header=None, engine="python", sep=",")
            if df.shape[1] == 0:
                raise RuntimeError("CSV 无列数据")
            s = pd.to_numeric(df.iloc[:, -1], errors="coerce")
            s = s.dropna()
            arr = s.to_numpy(dtype=np.float32)
            if arr.size == 0:
                raise RuntimeError("CSV 无有效数值")
            return arr
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"读取CSV失败（尝试utf-8-sig与gbk）：{path}；原因：{last_err}")

def standardize(x: np.ndarray, method: str = "zscore") -> np.ndarray:
    if x is None or x.size == 0:
        raise RuntimeError("标准化输入为空")
    method = (method or "zscore").lower()
    if method == "none":
        return x.astype(np.float32, copy=False)
    if method == "zscore":
        m = float(np.mean(x))
        s = float(np.std(x))
        if s < 1e-8:
            return (x - m).astype(np.float32)
        return ((x - m) / (s + 1e-12)).astype(np.float32)
    if method == "minmax":
        xmin = float(np.min(x))
        xmax = float(np.max(x))
        if xmax - xmin < 1e-12:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - xmin) / (xmax - xmin)).astype(np.float32)
    raise RuntimeError(f"不支持的标准化方法: {method}")

class SpectrumDataset(Dataset):
    def __init__(self, files: List[str], labels: List[int], standardize_method: str = "zscore", transform=None):
        if len(files) != len(labels):
            raise RuntimeError("files 与 labels 数量不一致")
        self.files = list(files)
        self.labels = list(labels)
        self.standardize_method = standardize_method
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        y = int(self.labels[idx])
        x = read_spectrum_csv(path)
        x = standardize(x, self.standardize_method)
        if self.transform is not None:
            x = self.transform(x)
        if isinstance(x, np.ndarray):
            x_t = torch.from_numpy(x.astype(np.float32, copy=False))
        elif torch.is_tensor(x):
            x_t = x.to(dtype=torch.float32)
        else:
            raise RuntimeError("transform 输出类型不支持，应返回 numpy.ndarray 或 torch.Tensor")
        return x_t, y

def collate_pad_1d(batch: List[Tuple[torch.Tensor, int]]):
    if len(batch) == 0:
        return torch.empty(0, 1, 0, dtype=torch.float32), torch.empty(0, dtype=torch.long)
    lengths = [int(x.shape[0]) for x, _ in batch]
    Lmax = max(lengths)
    B = len(batch)
    xs = torch.zeros(B, Lmax, dtype=torch.float32)
    ys = torch.empty(B, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        l = x.shape[0]
        xs[i, :l] = x.to(dtype=torch.float32)
        ys[i] = int(y)
    xs = xs.unsqueeze(1)
    return xs, ys

def _list_all_subdirs(root_dir: str) -> List[str]:
    dirs: List[str] = []
    for d, subdirs, _ in os.walk(root_dir):
        for s in subdirs:
            dirs.append(os.path.join(d, s))
    return dirs

def _collect_csv_files_from_dirs(dirs: List[str]) -> List[str]:
    files: List[str] = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for r, _, fs in os.walk(d):
            for f in fs:
                if f.lower().endswith(".csv"):
                    files.append(os.path.join(r, f))
    files.sort()
    return files

def _find_dirs_by_keywords(root_dir: str, include_any: List[str], exclude_any: Optional[List[str]] = None) -> List[str]:
    if not os.path.isdir(root_dir):
        return []
    dirs = _list_all_subdirs(root_dir)
    out: List[str] = []
    for d in dirs:
        name = os.path.basename(d)
        inc = any(k in name for k in include_any) if include_any else True
        exc = any(k in name for k in (exclude_any or [])) if exclude_any else False
        if inc and not exc:
            out.append(d)
    base = os.path.basename(root_dir)
    if (include_any and any(k in base for k in include_any)) and not any(k in base for k in (exclude_any or [])):
        out.append(root_dir)
    out = sorted(set(out))
    return out

def _find_pure_dirs(primary_root: str, global_root: str) -> List[str]:
    pure = _find_dirs_by_keywords(primary_root, include_any=["纯", "无污染"], exclude_any=["PP", "PE"])
    if pure:
        return pure
    pure = _find_dirs_by_keywords(global_root, include_any=["纯", "无污染"], exclude_any=["PP", "PE"])
    if pure:
        return pure
    raise RuntimeError("未找到\"无污染\"目录，请在对应二元根目录或 dataset 下添加或指明该目录")

def _require_and_split(files: List[str], required_total: int, train_n: int, seed: int) -> Tuple[List[str], List[str]]:
    if len(files) < required_total:
        raise RuntimeError(f"样本数不足：需要 {required_total}，实际 {len(files)}。请检查数据目录。")
    rnd = random.Random(int(seed))
    f = list(files)
    rnd.shuffle(f)
    f = f[:required_total]
    train = f[:train_n]
    val = f[train_n:required_total]
    return train, val

def _build_loaders_from_class_lists(
    class_to_files_train: Dict[int, List[str]],
    class_to_files_val: Dict[int, List[str]],
    batch_size: int,
    standardize_method: str,
    seed: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    train_files: List[str] = []
    train_labels: List[int] = []
    val_files: List[str] = []
    val_labels: List[int] = []
    for c in sorted(class_to_files_train.keys()):
        fs = class_to_files_train[c]
        train_files.extend(fs)
        train_labels.extend([c] * len(fs))
    for c in sorted(class_to_files_val.keys()):
        fs = class_to_files_val[c]
        val_files.extend(fs)
        val_labels.extend([c] * len(fs))

    train_ds = SpectrumDataset(train_files, train_labels, standardize_method=standardize_method)
    val_ds = SpectrumDataset(val_files, val_labels, standardize_method=standardize_method)

    pin = torch.cuda.is_available()
    g = torch.Generator()
    g.manual_seed(int(seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(num_workers or 0),
        pin_memory=pin,
        collate_fn=collate_pad_1d,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(num_workers or 0),
        pin_memory=pin,
        collate_fn=collate_pad_1d,
    )
    return train_loader, val_loader

def _get_binary_mix_loaders(
    root: str,
    mix_name: str,
    batch_size: int,
    standardize_method: str,
    seed: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, Dict]:
    primary_root = os.path.join(root, mix_name)
    if not os.path.isdir(primary_root):
        raise RuntimeError(f"找不到数据根目录: {primary_root}")
    slight_dirs = _find_dirs_by_keywords(primary_root, include_any=["轻微浓度"])
    severe_dirs = _find_dirs_by_keywords(primary_root, include_any=["严重浓度"])
    pure_dirs = _find_pure_dirs(primary_root, root)
    if not slight_dirs:
        raise RuntimeError(f"未找到\"轻微浓度\"目录于: {primary_root}")
    if not severe_dirs:
        raise RuntimeError(f"未找到\"严重浓度\"目录于: {primary_root}")
    slight_files = _collect_csv_files_from_dirs(slight_dirs)
    severe_files = _collect_csv_files_from_dirs(severe_dirs)
    pure_files = _collect_csv_files_from_dirs(pure_dirs)
    required = 30
    train_n = 22
    slight_tr, slight_va = _require_and_split(slight_files, required, train_n, seed)
    severe_tr, severe_va = _require_and_split(severe_files, required, train_n, seed)
    pure_tr, pure_va = _require_and_split(pure_files, required, train_n, seed)
    class_to_train = {0: pure_tr, 1: slight_tr, 2: severe_tr}
    class_to_val = {0: pure_va, 1: slight_va, 2: severe_va}
    train_loader, val_loader = _build_loaders_from_class_lists(
        class_to_train, class_to_val, batch_size, standardize_method, seed, num_workers
    )
    meta = {
        "class_names": {0: "无污染", 1: "轻微浓度", 2: "严重浓度"},
        "train_counts": {k: len(v) for k, v in class_to_train.items()},
        "val_counts": {k: len(v) for k, v in class_to_val.items()},
    }
    return train_loader, val_loader, meta

def get_pp_three_class_loaders(
    root: str = "dataset",
    batch_size: int = 32,
    standardize_method: str = "zscore",
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Dict]:
    return _get_binary_mix_loaders(root, "PP+淀粉", batch_size, standardize_method, seed, num_workers)

def get_pe_three_class_loaders(
    root: str = "dataset",
    batch_size: int = 32,
    standardize_method: str = "zscore",
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Dict]:
    return _get_binary_mix_loaders(root, "PE+淀粉", batch_size, standardize_method, seed, num_workers)

def _get_ternary_subtask_loaders(
    root: str,
    sub: str,
    batch_size: int,
    standardize_method: str,
    seed: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, Dict]:
    root_dir = os.path.join(root, "PP+PE+淀粉")
    if not os.path.isdir(root_dir):
        raise RuntimeError(f"找不到数据根目录: {root_dir}")
    all_dirs = _list_all_subdirs(root_dir)
    if sub.upper() == "PP":
        slight_dirs = [d for d in all_dirs if "轻微PP" in os.path.basename(d)]
        severe_dirs = [d for d in all_dirs if "严重PP" in os.path.basename(d)]
        name_slight = "PP轻微(原1)"
        name_severe = "PP严重(原2)"
    elif sub.upper() == "PE":
        slight_dirs = [d for d in all_dirs if "轻微PE" in os.path.basename(d)]
        severe_dirs = [d for d in all_dirs if "严重PE" in os.path.basename(d)]
        name_slight = "PE轻微(原1)"
        name_severe = "PE严重(原2)"
    else:
        raise RuntimeError("sub 仅支持 'PP' 或 'PE'")
    if not slight_dirs or not severe_dirs:
        raise RuntimeError(f"未找到包含 '{'轻微'+sub}' 或 '{'严重'+sub}' 的目录，请检查 {root_dir}")
    slight_files = _collect_csv_files_from_dirs(slight_dirs)
    severe_files = _collect_csv_files_from_dirs(severe_dirs)
    required = 50
    train_n = 40
    slight_tr, slight_va = _require_and_split(slight_files, required, train_n, seed)
    severe_tr, severe_va = _require_and_split(severe_files, required, train_n, seed)
    def map_bin(orig: int) -> int:
        return 0 if orig == 1 else 1
    train_files = slight_tr + severe_tr
    train_labels = [map_bin(1)] * len(slight_tr) + [map_bin(2)] * len(severe_tr)
    val_files = slight_va + severe_va
    val_labels = [map_bin(1)] * len(slight_va) + [map_bin(2)] * len(severe_va)
    train_ds = SpectrumDataset(train_files, train_labels, standardize_method=standardize_method)
    val_ds = SpectrumDataset(val_files, val_labels, standardize_method=standardize_method)
    pin = torch.cuda.is_available()
    g = torch.Generator()
    g.manual_seed(int(seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(num_workers or 0),
        pin_memory=pin,
        collate_fn=collate_pad_1d,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(num_workers or 0),
        pin_memory=pin,
        collate_fn=collate_pad_1d,
    )
    meta = {
        "class_names": {0: name_slight, 1: name_severe},
        "label_inverse_map": {0: 1, 1: 2},
        "train_counts": {1: len(slight_tr), 2: len(severe_tr)},
        "val_counts": {1: len(slight_va), 2: len(severe_va)},
    }
    return train_loader, val_loader, meta

def get_pp_two_class_loaders(
    root: str = "dataset",
    batch_size: int = 32,
    standardize_method: str = "zscore",
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Dict]:
    return _get_ternary_subtask_loaders(root, "PP", batch_size, standardize_method, seed, num_workers)

def get_pe_two_class_loaders(
    root: str = "dataset",
    batch_size: int = 32,
    standardize_method: str = "zscore",
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Dict]:
    return _get_ternary_subtask_loaders(root, "PE", batch_size, standardize_method, seed, num_workers)

if __name__ == "__main__":
    print("=== 数据加载模块最小示例 ===")
    try:
        tr, va, meta = get_pp_three_class_loaders()
        print(f"PP+淀粉 三分类: 训练批数={len(tr)}, 验证批数={len(va)}, meta={meta}")
        xb, yb = next(iter(tr))
        print(f"一个batch: x形状={xb.shape}, y形状={yb.shape}")
    except Exception as e:
        print(f"[提示] PP+淀粉示例未运行：{e}")
    try:
        tr, va, meta = get_pp_two_class_loaders()
        print(f"三元混合 PP 子任务: 训练批数={len(tr)}, 验证批数={len(va)}, meta={meta}")
        xb, yb = next(iter(tr))
        print(f"一个batch: x形状={xb.shape}, y形状={yb.shape}")
    except Exception as e:
        print(f"[提示] 三元混合 PP 子任务示例未运行：{e}")

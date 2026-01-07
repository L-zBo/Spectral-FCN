# -*- coding: utf-8 -*-
"""
Spectral-FCN 公共工具模块

提供数据读取、标准化、文件收集等公共功能，避免代码重复。
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
OUTPUT_ROOT = PROJECT_ROOT / "output"
MODEL_DATA_ROOT = OUTPUT_ROOT / "model_data"


def get_project_root() -> Path:
    """获取项目根目录"""
    return PROJECT_ROOT


def get_dataset_root() -> Path:
    """获取数据集根目录"""
    return DATASET_ROOT


def get_output_root() -> Path:
    """获取输出根目录"""
    return OUTPUT_ROOT


def read_spectrum_csv(path: str) -> np.ndarray:
    """
    读取光谱 CSV 文件

    Args:
        path: CSV 文件路径

    Returns:
        光谱数据数组

    Raises:
        RuntimeError: 无法读取文件时抛出
    """
    if not os.path.isfile(path):
        raise RuntimeError(f"找不到文件: {path}")

    last_err: Optional[Exception] = None
    for enc in ("utf-8-sig", "gbk", "utf-8"):
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
    """
    标准化光谱数据

    Args:
        x: 输入数组
        method: 标准化方法 ("zscore", "minmax", "none")

    Returns:
        标准化后的数组
    """
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


def collect_csv_files(directory: str) -> List[str]:
    """
    收集目录下所有 CSV 文件

    Args:
        directory: 目录路径

    Returns:
        CSV 文件路径列表（已排序）
    """
    files = []
    if os.path.exists(directory):
        for root, _, filenames in os.walk(directory):
            for f in filenames:
                if f.lower().endswith('.csv'):
                    files.append(os.path.join(root, f))
    return sorted(files)


def split_files(files: List[str], train_ratio: float = 0.75,
                seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    随机分割文件列表为训练集和验证集

    Args:
        files: 文件列表
        train_ratio: 训练集比例
        seed: 随机种子

    Returns:
        (训练文件列表, 验证文件列表)
    """
    rnd = random.Random(seed)
    f = list(files)
    rnd.shuffle(f)
    train_n = int(len(f) * train_ratio)
    return f[:train_n], f[train_n:]


def stack_pad(arrs: List[np.ndarray], fixed_len: Optional[int] = None) -> np.ndarray:
    """
    将不同长度的数组填充并堆叠

    Args:
        arrs: 数组列表
        fixed_len: 固定长度（可选）

    Returns:
        堆叠后的 2D 数组
    """
    if len(arrs) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    max_len = fixed_len if fixed_len else max(a.shape[0] for a in arrs)
    padded = []

    for a in arrs:
        if a.shape[0] < max_len:
            out = np.zeros(max_len, dtype=np.float32)
            out[:a.shape[0]] = a
            padded.append(out)
        else:
            padded.append(a[:max_len])

    return np.stack(padded, axis=0)


def set_seed_all(seed: int) -> None:
    """
    设置所有随机种子以确保可重复性

    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """获取可用的计算设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 任务配置
TASKS_CONFIG = {
    "PP_Starch": {
        "name": "PP+Starch",
        "data_dir": "PP+淀粉",
        "model_dir": "PP_Starch",
        "num_classes": 3,
        "class_names": ["Pollution-free", "Slight Conc.", "Severe Conc."],
        "class_names_cn": ["无污染", "轻微浓度", "严重浓度"],
        "subdirs": ["无污染", "轻微浓度", "严重浓度"]
    },
    "PE_Starch": {
        "name": "PE+Starch",
        "data_dir": "PE+淀粉",
        "model_dir": "PE_Starch",
        "num_classes": 3,
        "class_names": ["Pollution-free", "Slight Conc.", "Severe Conc."],
        "class_names_cn": ["无污染", "轻微浓度", "严重浓度"],
        "subdirs": ["无污染", "轻微浓度", "严重浓度"]
    },
    "PP_PE_Starch": {
        "name": "PP+PE+Starch",
        "data_dir": "PP+PE+淀粉",
        "model_dir": "PP_PE_Starch",
        "num_classes": 3,
        "class_names": ["Pollution-free", "Slight Conc.", "Severe Conc."],
        "class_names_cn": ["无污染", "轻微浓度", "严重浓度"],
        "subdirs": ["无污染", "轻微PP+轻微PE+淀粉", "严重PP+严重PE+淀粉"]
    }
}


def load_task_data(task_key: str, train_ratio: float = 0.75,
                   seed: int = 42) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    加载指定任务的数据

    Args:
        task_key: 任务键名 (PP_Starch, PE_Starch, PP_PE_Starch)
        train_ratio: 训练集比例
        seed: 随机种子

    Returns:
        (训练文件, 训练标签, 验证文件, 验证标签)
    """
    if task_key not in TASKS_CONFIG:
        raise ValueError(f"未知任务: {task_key}")

    config = TASKS_CONFIG[task_key]
    base_dir = DATASET_ROOT / config["data_dir"]

    all_train_files = []
    all_train_labels = []
    all_val_files = []
    all_val_labels = []

    for label_idx, subdir in enumerate(config["subdirs"]):
        subdir_path = base_dir / subdir
        files = collect_csv_files(str(subdir_path))

        if not files:
            print(f"  警告: {subdir_path} 下没有找到 CSV 文件")
            continue

        train_files, val_files = split_files(files, train_ratio, seed)

        all_train_files.extend(train_files)
        all_train_labels.extend([label_idx] * len(train_files))
        all_val_files.extend(val_files)
        all_val_labels.extend([label_idx] * len(val_files))

    return all_train_files, all_train_labels, all_val_files, all_val_labels


def load_spectra_from_files(files: List[str],
                            standardize_method: str = "zscore") -> List[np.ndarray]:
    """
    从文件列表加载并标准化光谱数据

    Args:
        files: 文件路径列表
        standardize_method: 标准化方法

    Returns:
        标准化后的光谱数据列表
    """
    spectra = []
    for f in files:
        spec = read_spectrum_csv(f)
        spec = standardize(spec, standardize_method)
        spectra.append(spec)
    return spectra


if __name__ == "__main__":
    print("=== 工具模块测试 ===")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"数据集目录: {DATASET_ROOT}")
    print(f"输出目录: {OUTPUT_ROOT}")
    print(f"计算设备: {get_device()}")

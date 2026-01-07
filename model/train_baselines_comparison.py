# -*- coding: utf-8 -*-
"""
传统机器学习模型对比实验

对比 KNN、SVM、Random Forest 等传统方法与 FCN 的性能差异。
"""

import os
import sys
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "data_processing"))

from utils import (
    get_dataset_root, get_output_root, TASKS_CONFIG,
    read_spectrum_csv, standardize, collect_csv_files,
    split_files, stack_pad, set_seed_all
)

SEED = 42


def prepare_data(task_key: str, train_ratio: float = 0.75):
    """
    准备指定任务的训练和验证数据

    Args:
        task_key: 任务键名
        train_ratio: 训练集比例

    Returns:
        (X_train, y_train, X_val, y_val)
    """
    if task_key not in TASKS_CONFIG:
        return None, None, None, None

    config = TASKS_CONFIG[task_key]
    base_dir = get_dataset_root() / config["data_dir"]

    train_arrs = []
    train_labels = []
    val_arrs = []
    val_labels = []

    for label_idx, subdir in enumerate(config["subdirs"]):
        subdir_path = base_dir / subdir
        files = collect_csv_files(str(subdir_path))

        if not files:
            print(f"  警告: {subdir_path} 下没有找到 CSV 文件")
            continue

        # 使用随机分割
        train_files, val_files = split_files(files, train_ratio, SEED)

        for f in train_files:
            try:
                spec = read_spectrum_csv(f)
                spec = standardize(spec, "minmax")
                train_arrs.append(spec)
                train_labels.append(label_idx)
            except Exception as e:
                print(f"  警告: 读取 {f} 失败: {e}")
                continue

        for f in val_files:
            try:
                spec = read_spectrum_csv(f)
                spec = standardize(spec, "minmax")
                val_arrs.append(spec)
                val_labels.append(label_idx)
            except Exception as e:
                print(f"  警告: 读取 {f} 失败: {e}")
                continue

    if not train_arrs or not val_arrs:
        return None, None, None, None

    # 填充到相同长度
    all_arrs = train_arrs + val_arrs
    max_len = max(a.shape[0] for a in all_arrs)

    X_train = stack_pad(train_arrs, max_len)
    X_val = stack_pad(val_arrs, max_len)
    y_train = np.array(train_labels)
    y_val = np.array(val_labels)

    return X_train, y_train, X_val, y_val


def evaluate_model(model, X_train, y_train, X_val, y_val):
    """
    训练并评估模型

    Args:
        model: sklearn 模型
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征
        y_val: 验证标签

    Returns:
        评估指标字典
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    return {
        "acc": float(accuracy_score(y_val, y_pred)),
        "f1_macro": float(f1_score(y_val, y_pred, average='macro')),
        "recall_macro": float(recall_score(y_val, y_pred, average='macro'))
    }


def main():
    print("=" * 60)
    print("传统机器学习模型对比实验")
    print("=" * 60)

    set_seed_all(SEED)

    # 任务映射
    task_mapping = {
        "pp_three_class": "PP_Starch",
        "pe_three_class": "PE_Starch",
        "pp_pe_three_class": "PP_PE_Starch"
    }

    all_results = {}

    for task_name, task_key in task_mapping.items():
        config = TASKS_CONFIG[task_key]
        print(f"\n{'=' * 50}")
        print(f"任务: {config['name']}")
        print("=" * 50)

        X_train, y_train, X_val, y_val = prepare_data(task_key)

        if X_train is None or len(X_train) == 0:
            print(f"跳过 {task_name} - 数据不可用")
            continue

        print(f"训练样本: {len(X_train)}, 验证样本: {len(X_val)}")
        print(f"特征维度: {X_train.shape[1]}")

        # KNN
        print("\n训练 KNN...")
        knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
        knn_results = evaluate_model(knn, X_train, y_train, X_val, y_val)
        print(f"  KNN: acc={knn_results['acc']:.4f}")

        # SVM
        print("训练 SVM...")
        svm = SVC(kernel='rbf', C=1.0, gamma='scale')
        svm_results = evaluate_model(svm, X_train, y_train, X_val, y_val)
        print(f"  SVM(RBF): acc={svm_results['acc']:.4f}")

        # Random Forest
        print("训练 Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED)
        rf_results = evaluate_model(rf, X_train, y_train, X_val, y_val)
        print(f"  Random Forest: acc={rf_results['acc']:.4f}")

        all_results[task_name] = {
            "knn": knn_results,
            "svm_rbf": svm_results,
            "random_forest": rf_results
        }

    # 保存结果
    output_dir = get_output_root()
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / "baseline_comparison_results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 打印汇总
    print(f"\n{'=' * 60}")
    print("汇总 - 基准模型结果")
    print("=" * 60)

    for task_name, results in all_results.items():
        task_key = task_mapping[task_name]
        config = TASKS_CONFIG[task_key]
        print(f"\n{config['name']}:")
        for model_name, metrics in results.items():
            display_name = model_name.upper().replace("_", " ")
            if model_name == "svm_rbf":
                display_name = "SVM(RBF)"
            print(f"  {display_name:15s}: acc={metrics['acc']:.4f}, "
                  f"f1={metrics['f1_macro']:.4f}, recall={metrics['recall_macro']:.4f}")

    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()

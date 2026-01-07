# -*- coding: utf-8 -*-
"""
混淆矩阵可视化模块

使用训练好的模型在验证集上进行真实预测，生成混淆矩阵。
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "model"))
sys.path.insert(0, str(project_root / "data_processing"))

from FCN import FaultClassificationNetwork
from utils import (
    TASKS_CONFIG, get_device, read_spectrum_csv, standardize,
    collect_csv_files, get_output_root, get_dataset_root, split_files
)


def load_validation_data(task_key: str, train_ratio: float = 0.75, seed: int = 42):
    """
    加载验证集数据

    Args:
        task_key: 任务键名
        train_ratio: 训练集比例（验证集为剩余部分）
        seed: 随机种子

    Returns:
        (光谱数据列表, 标签列表)
    """
    config = TASKS_CONFIG[task_key]
    base_dir = get_dataset_root() / config["data_dir"]

    val_spectra = []
    val_labels = []

    for label_idx, subdir in enumerate(config["subdirs"]):
        subdir_path = base_dir / subdir
        files = collect_csv_files(str(subdir_path))

        if not files:
            print(f"  警告: {subdir_path} 下没有找到 CSV 文件")
            continue

        # 使用相同的随机种子分割，确保与训练时一致
        _, val_files = split_files(files, train_ratio, seed)

        for f in val_files:
            try:
                spec = read_spectrum_csv(f)
                spec = standardize(spec, "zscore")
                val_spectra.append(spec)
                val_labels.append(label_idx)
            except Exception as e:
                print(f"  警告: 读取 {f} 失败: {e}")
                continue

    return val_spectra, val_labels


def evaluate_model(model_dir: str, spectra: list, labels: list, num_classes: int):
    """
    使用真实模型进行预测

    Args:
        model_dir: 模型目录
        spectra: 光谱数据列表
        labels: 真实标签列表
        num_classes: 类别数

    Returns:
        (预测标签列表, 真实标签列表, 准确率)
    """
    device = get_device()

    model = FaultClassificationNetwork(num_classes=num_classes)
    encoder_path = os.path.join(model_dir, "encoder.pth")
    classifier_path = os.path.join(model_dir, "classifier.pth")

    if not os.path.exists(encoder_path):
        print(f"  错误: 模型文件不存在 {encoder_path}")
        return None, None, None

    model.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    model.classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    model = model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for spec in spectra:
            x = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).item()
            predictions.append(pred)

    accuracy = accuracy_score(labels, predictions)

    return predictions, labels, accuracy


def plot_confusion_matrix(task_key: str, output_dir: str):
    """
    为指定任务生成混淆矩阵

    Args:
        task_key: 任务键名
        output_dir: 输出目录
    """
    config = TASKS_CONFIG[task_key]
    print(f"\n生成 {config['name']} 的混淆矩阵...")

    # 加载验证数据
    spectra, labels = load_validation_data(task_key)
    if not spectra:
        print(f"  跳过: 没有可用数据")
        return

    print(f"  验证集样本数: {len(spectra)}")

    # 获取模型路径
    model_dir = get_output_root() / "model_data" / config["model_dir"] / "checkpoints" / "best"

    # 使用真实模型预测
    predictions, true_labels, accuracy = evaluate_model(
        str(model_dir), spectra, labels, config["num_classes"]
    )

    if predictions is None:
        print(f"  跳过: 模型文件不存在")
        return

    print(f"  真实准确率: {accuracy:.4f}")

    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, predictions)

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=config['class_names'],
        yticklabels=config['class_names'],
        cbar=True,
        square=True,
        linewidths=1,
        linecolor='gray',
        ax=ax,
        annot_kws={'size': 14, 'weight': 'bold'}
    )

    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title(
        f'{config["name"]} - Confusion Matrix\nAccuracy: {accuracy:.4f}',
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()

    output_path = os.path.join(output_dir, f'{task_key}_confusion_matrix.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  已保存: {output_path}")


def plot_normalized_confusion_matrix(task_key: str, output_dir: str):
    """
    为指定任务生成归一化混淆矩阵

    Args:
        task_key: 任务键名
        output_dir: 输出目录
    """
    config = TASKS_CONFIG[task_key]
    print(f"\n生成 {config['name']} 的归一化混淆矩阵...")

    # 加载验证数据
    spectra, labels = load_validation_data(task_key)
    if not spectra:
        print(f"  跳过: 没有可用数据")
        return

    # 获取模型路径
    model_dir = get_output_root() / "model_data" / config["model_dir"] / "checkpoints" / "best"

    # 使用真实模型预测
    predictions, true_labels, accuracy = evaluate_model(
        str(model_dir), spectra, labels, config["num_classes"]
    )

    if predictions is None:
        print(f"  跳过: 模型文件不存在")
        return

    # 计算归一化混淆矩阵
    cm = confusion_matrix(true_labels, predictions, normalize='true')

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=config['class_names'],
        yticklabels=config['class_names'],
        cbar=True,
        square=True,
        linewidths=1,
        linecolor='gray',
        ax=ax,
        annot_kws={'size': 12, 'weight': 'bold'},
        vmin=0,
        vmax=1
    )

    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title(
        f'{config["name"]} - Normalized Confusion Matrix\nAccuracy: {accuracy:.4f}',
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()

    output_path = os.path.join(output_dir, f'{task_key}_confusion_matrix_normalized.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  已保存: {output_path}")


def main():
    print("=" * 60)
    print("生成混淆矩阵（使用真实模型预测）")
    print("=" * 60)

    confusion_matrix_dir = get_output_root() / "混淆矩阵"
    os.makedirs(confusion_matrix_dir, exist_ok=True)

    for task_key in TASKS_CONFIG.keys():
        plot_confusion_matrix(task_key, str(confusion_matrix_dir))
        plot_normalized_confusion_matrix(task_key, str(confusion_matrix_dir))

    print("\n" + "=" * 60)
    print(f"所有混淆矩阵已保存到: {confusion_matrix_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

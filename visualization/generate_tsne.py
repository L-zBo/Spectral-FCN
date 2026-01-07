# -*- coding: utf-8 -*-
"""
t-SNE 特征可视化模块

使用真实的 t-SNE 降维算法对模型特征进行可视化。
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE

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
    collect_csv_files, get_output_root, get_dataset_root
)

# 配色方案
CLASS_COLORS = ['#2ecc71', '#3498db', '#e74c3c']


def load_all_data(task_key: str):
    """
    加载指定任务的所有数据

    Args:
        task_key: 任务键名

    Returns:
        (光谱数据列表, 标签列表)
    """
    config = TASKS_CONFIG[task_key]
    base_dir = get_dataset_root() / config["data_dir"]

    all_spectra = []
    all_labels = []

    for label_idx, subdir in enumerate(config["subdirs"]):
        subdir_path = base_dir / subdir
        files = collect_csv_files(str(subdir_path))

        for f in files:
            try:
                spec = read_spectrum_csv(f)
                spec = standardize(spec, "zscore")
                all_spectra.append(spec)
                all_labels.append(label_idx)
            except Exception as e:
                print(f"  警告: 读取 {f} 失败: {e}")
                continue

    return all_spectra, all_labels


def extract_features(model_dir: str, spectra: list, num_classes: int):
    """
    使用训练好的模型提取特征

    Args:
        model_dir: 模型目录
        spectra: 光谱数据列表
        num_classes: 类别数

    Returns:
        特征数组
    """
    device = get_device()

    model = FaultClassificationNetwork(num_classes=num_classes)
    encoder_path = os.path.join(model_dir, "encoder.pth")
    classifier_path = os.path.join(model_dir, "classifier.pth")

    if not os.path.exists(encoder_path):
        print(f"  警告: 模型文件不存在 {encoder_path}")
        return None

    model.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    model.classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_features = []
    with torch.no_grad():
        for spec in spectra:
            x = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).to(device)
            features = model.encoder(x)
            features_flat = features.view(features.size(0), -1).cpu().numpy()[0]
            all_features.append(features_flat)

    return np.array(all_features)


def compute_tsne(features: np.ndarray, perplexity: int = 30,
                 n_iter: int = 1000, seed: int = 42) -> np.ndarray:
    """
    使用真实的 t-SNE 算法进行降维

    Args:
        features: 高维特征数组
        perplexity: t-SNE perplexity 参数
        n_iter: 迭代次数
        seed: 随机种子

    Returns:
        降维后的 2D 坐标
    """
    print(f"  执行 t-SNE 降维 (perplexity={perplexity}, n_iter={n_iter})...")

    # 根据样本数调整 perplexity
    n_samples = features.shape[0]
    adjusted_perplexity = min(perplexity, n_samples - 1)

    tsne = TSNE(
        n_components=2,
        perplexity=adjusted_perplexity,
        n_iter=n_iter,
        random_state=seed,
        init='pca',
        learning_rate='auto'
    )

    features_2d = tsne.fit_transform(features)
    print(f"  t-SNE 降维完成，输出形状: {features_2d.shape}")

    return features_2d


def plot_tsne_for_task(task_key: str, output_dir: str):
    """
    为指定任务生成 t-SNE 可视化

    Args:
        task_key: 任务键名
        output_dir: 输出目录
    """
    config = TASKS_CONFIG[task_key]
    print(f"\n生成 {config['name']} 的 t-SNE 可视化...")

    # 加载数据
    spectra, labels = load_all_data(task_key)
    if not spectra:
        print(f"  跳过 {task_key} - 没有可用数据")
        return

    print(f"  加载了 {len(spectra)} 个样本")

    # 提取特征
    model_dir = get_output_root() / "model_data" / config["model_dir"] / "checkpoints" / "best"
    features = extract_features(str(model_dir), spectra, config["num_classes"])

    if features is None:
        # 如果没有模型，直接用原始光谱数据
        print("  模型不存在，使用原始光谱数据进行 t-SNE")
        max_len = max(s.shape[0] for s in spectra)
        features = np.zeros((len(spectra), max_len), dtype=np.float32)
        for i, s in enumerate(spectra):
            features[i, :s.shape[0]] = s

    # 执行真实的 t-SNE 降维
    features_2d = compute_tsne(features)

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))

    labels_array = np.array(labels)
    for class_idx in range(config["num_classes"]):
        mask = labels_array == class_idx
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=CLASS_COLORS[class_idx],
            label=config["class_names"][class_idx],
            s=100,
            alpha=0.8,
            edgecolors='white',
            linewidth=0.8
        )

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title(f'{config["name"]} - t-SNE Feature Visualization', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()

    output_path = os.path.join(output_dir, f'{task_key}_tsne.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  已保存: {output_path}")


def plot_all_tasks_tsne(output_dir: str):
    """
    生成所有任务的 t-SNE 对比图

    Args:
        output_dir: 输出目录
    """
    print("\n生成所有任务的 t-SNE 对比图...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (task_key, config) in enumerate(TASKS_CONFIG.items()):
        ax = axes[idx]

        # 加载数据
        spectra, labels = load_all_data(task_key)
        if not spectra:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(config['name'], fontsize=12, fontweight='bold')
            continue

        # 提取特征
        model_dir = get_output_root() / "model_data" / config["model_dir"] / "checkpoints" / "best"
        features = extract_features(str(model_dir), spectra, config["num_classes"])

        if features is None:
            max_len = max(s.shape[0] for s in spectra)
            features = np.zeros((len(spectra), max_len), dtype=np.float32)
            for i, s in enumerate(spectra):
                features[i, :s.shape[0]] = s

        # 执行 t-SNE
        features_2d = compute_tsne(features)

        # 绘图
        labels_array = np.array(labels)
        for class_idx in range(config["num_classes"]):
            mask = labels_array == class_idx
            ax.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=CLASS_COLORS[class_idx],
                label=config["class_names"][class_idx],
                s=80,
                alpha=0.8,
                edgecolors='white',
                linewidth=0.6
            )

        ax.set_xlabel('t-SNE Dim 1', fontsize=10, fontweight='bold')
        ax.set_ylabel('t-SNE Dim 2', fontsize=10, fontweight='bold')
        ax.set_title(config['name'], fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3, linestyle='--')

    plt.suptitle('t-SNE Feature Visualization Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'all_tasks_tsne_comparison.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  已保存: {output_path}")


def main():
    print("=" * 60)
    print("生成 t-SNE 可视化（使用真实降维算法）")
    print("=" * 60)

    tsne_dir = get_output_root() / "t-SNE降维"
    os.makedirs(tsne_dir, exist_ok=True)

    for task_key in TASKS_CONFIG.keys():
        plot_tsne_for_task(task_key, str(tsne_dir))

    plot_all_tasks_tsne(str(tsne_dir))

    print("\n" + "=" * 60)
    print(f"所有 t-SNE 可视化已保存到: {tsne_dir}")
    print("生成的文件:")
    for task_key in TASKS_CONFIG.keys():
        print(f"  - {task_key}_tsne.png")
    print("  - all_tasks_tsne_comparison.png")
    print("=" * 60)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
SHAP 值可视化模块

使用 SHAP 库计算真实的 Shapley 值，解释模型决策。

注意：需要安装 shap 库: pip install shap
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib

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
    collect_csv_files, get_output_root, get_dataset_root,
    split_files, stack_pad
)

# 尝试导入 shap
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("警告: shap 库未安装，将使用基于梯度的近似方法")
    print("安装方法: pip install shap")


def load_data_for_shap(task_key: str, n_samples: int = 50, seed: int = 42):
    """
    加载用于 SHAP 分析的数据

    Args:
        task_key: 任务键名
        n_samples: 每个类别的样本数
        seed: 随机种子

    Returns:
        (特征数组, 标签数组, 特征长度)
    """
    config = TASKS_CONFIG[task_key]
    base_dir = get_dataset_root() / config["data_dir"]

    all_spectra = []
    all_labels = []

    for label_idx, subdir in enumerate(config["subdirs"]):
        subdir_path = base_dir / subdir
        files = collect_csv_files(str(subdir_path))

        if not files:
            continue

        # 随机选择样本
        np.random.seed(seed + label_idx)
        selected = np.random.choice(len(files), min(n_samples, len(files)), replace=False)

        for idx in selected:
            try:
                spec = read_spectrum_csv(files[idx])
                spec = standardize(spec, "zscore")
                all_spectra.append(spec)
                all_labels.append(label_idx)
            except Exception as e:
                continue

    if not all_spectra:
        return None, None, None

    # 填充到相同长度
    X = stack_pad(all_spectra)
    y = np.array(all_labels)

    return X, y, X.shape[1]


def compute_gradient_importance(model, X, device):
    """
    使用梯度计算特征重要性（当 SHAP 不可用时的替代方案）

    Args:
        model: PyTorch 模型
        X: 输入数据
        device: 计算设备

    Returns:
        特征重要性数组
    """
    model.eval()

    X_tensor = torch.from_numpy(X).float().unsqueeze(1).to(device)
    X_tensor.requires_grad = True

    # 前向传播
    outputs = model(X_tensor)

    # 对预测类别的输出求梯度
    pred_classes = outputs.argmax(dim=1)
    selected_outputs = outputs[range(len(outputs)), pred_classes]

    # 反向传播
    model.zero_grad()
    selected_outputs.sum().backward()

    # 获取梯度的绝对值作为重要性
    gradients = X_tensor.grad.abs().squeeze(1).cpu().numpy()

    # 平均所有样本的梯度
    importance = gradients.mean(axis=0)

    return importance


def compute_shap_values(model, X_background, X_explain, device):
    """
    使用 SHAP 库计算真实的 Shapley 值

    Args:
        model: PyTorch 模型
        X_background: 背景数据
        X_explain: 待解释数据
        device: 计算设备

    Returns:
        SHAP 值数组
    """
    if not SHAP_AVAILABLE:
        return None

    model.eval()

    # 创建模型包装函数
    def model_predict(x):
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).float().unsqueeze(1).to(device)
            outputs = model(x_tensor)
            return outputs.cpu().numpy()

    # 创建 SHAP 解释器
    explainer = shap.KernelExplainer(model_predict, X_background[:50])

    # 计算 SHAP 值
    shap_values = explainer.shap_values(X_explain[:20])

    return shap_values


def plot_feature_importance(task_key: str, output_dir: str):
    """
    绘制特征重要性图

    Args:
        task_key: 任务键名
        output_dir: 输出目录
    """
    config = TASKS_CONFIG[task_key]
    print(f"\n计算 {config['name']} 的特征重要性...")

    # 加载数据
    X, y, feature_len = load_data_for_shap(task_key)
    if X is None:
        print(f"  跳过: 没有可用数据")
        return

    print(f"  样本数: {X.shape[0]}, 特征维度: {X.shape[1]}")

    # 加载模型
    device = get_device()
    model_dir = get_output_root() / "model_data" / config["model_dir"] / "checkpoints" / "best"
    encoder_path = model_dir / "encoder.pth"
    classifier_path = model_dir / "classifier.pth"

    if not encoder_path.exists():
        print(f"  跳过: 模型文件不存在")
        return

    model = FaultClassificationNetwork(num_classes=config["num_classes"])
    model.encoder.load_state_dict(torch.load(str(encoder_path), map_location=device))
    model.classifier.load_state_dict(torch.load(str(classifier_path), map_location=device))
    model = model.to(device)

    # 计算特征重要性
    print("  计算基于梯度的特征重要性...")
    importance = compute_gradient_importance(model, X, device)

    # 选择 top-20 重要特征
    n_features = min(20, len(importance))
    top_indices = np.argsort(importance)[-n_features:][::-1]
    top_importance = importance[top_indices]

    # 创建波数标签（假设拉曼光谱范围）
    wavenumbers = np.linspace(400, 1800, len(importance))
    top_labels = [f'{int(wavenumbers[i])} cm⁻¹' for i in top_indices]

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, n_features))
    bars = ax.barh(range(n_features), top_importance[::-1],
                   color=colors[::-1], edgecolor='white', linewidth=0.5)

    ax.set_yticks(range(n_features))
    ax.set_yticklabels(top_labels[::-1], fontsize=10)
    ax.set_xlabel('Feature Importance (Gradient-based)', fontsize=12, fontweight='bold')
    ax.set_title(f'{config["name"]} - Feature Importance Analysis', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # 添加数值标签
    for bar, val in zip(bars, top_importance[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9)

    plt.tight_layout()

    output_path = os.path.join(output_dir, f'{task_key}_feature_importance.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  已保存: {output_path}")


def plot_shap_summary(task_key: str, output_dir: str):
    """
    绘制 SHAP 汇总图（如果 SHAP 库可用）

    Args:
        task_key: 任务键名
        output_dir: 输出目录
    """
    if not SHAP_AVAILABLE:
        print("  跳过 SHAP 汇总图: shap 库未安装")
        return

    config = TASKS_CONFIG[task_key]
    print(f"\n计算 {config['name']} 的 SHAP 值...")

    # 加载数据
    X, y, feature_len = load_data_for_shap(task_key, n_samples=30)
    if X is None:
        print(f"  跳过: 没有可用数据")
        return

    # 加载模型
    device = get_device()
    model_dir = get_output_root() / "model_data" / config["model_dir"] / "checkpoints" / "best"
    encoder_path = model_dir / "encoder.pth"
    classifier_path = model_dir / "classifier.pth"

    if not encoder_path.exists():
        print(f"  跳过: 模型文件不存在")
        return

    model = FaultClassificationNetwork(num_classes=config["num_classes"])
    model.encoder.load_state_dict(torch.load(str(encoder_path), map_location=device))
    model.classifier.load_state_dict(torch.load(str(classifier_path), map_location=device))
    model = model.to(device)

    # 计算 SHAP 值
    print("  计算 SHAP 值（这可能需要一些时间）...")
    shap_values = compute_shap_values(model, X[:20], X[20:40], device)

    if shap_values is None:
        print("  SHAP 计算失败")
        return

    # 创建波数标签
    wavenumbers = np.linspace(400, 1800, X.shape[1])
    feature_names = [f'{int(w)}' for w in wavenumbers]

    # 绘制 SHAP 汇总图
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X[20:40], feature_names=feature_names, show=False)
    plt.title(f'{config["name"]} - SHAP Summary Plot', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(output_dir, f'{task_key}_shap_summary.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  已保存: {output_path}")


def plot_spectral_importance_overlay(task_key: str, output_dir: str):
    """
    在光谱图上叠加显示特征重要性

    Args:
        task_key: 任务键名
        output_dir: 输出目录
    """
    config = TASKS_CONFIG[task_key]
    print(f"\n生成 {config['name']} 的光谱重要性叠加图...")

    # 加载数据
    X, y, feature_len = load_data_for_shap(task_key, n_samples=30)
    if X is None:
        print(f"  跳过: 没有可用数据")
        return

    # 加载模型
    device = get_device()
    model_dir = get_output_root() / "model_data" / config["model_dir"] / "checkpoints" / "best"
    encoder_path = model_dir / "encoder.pth"
    classifier_path = model_dir / "classifier.pth"

    if not encoder_path.exists():
        print(f"  跳过: 模型文件不存在")
        return

    model = FaultClassificationNetwork(num_classes=config["num_classes"])
    model.encoder.load_state_dict(torch.load(str(encoder_path), map_location=device))
    model.classifier.load_state_dict(torch.load(str(classifier_path), map_location=device))
    model = model.to(device)

    # 计算特征重要性
    importance = compute_gradient_importance(model, X, device)

    # 创建波数轴
    wavenumbers = np.linspace(400, 1800, len(importance))

    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # 上图：平均光谱
    mean_spectrum = X.mean(axis=0)
    ax1.plot(wavenumbers, mean_spectrum, 'b-', linewidth=1.5, label='Mean Spectrum')
    ax1.fill_between(wavenumbers, mean_spectrum - X.std(axis=0),
                     mean_spectrum + X.std(axis=0), alpha=0.3, color='blue')
    ax1.set_ylabel('Intensity (a.u.)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{config["name"]} - Raman Spectrum', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3, linestyle='--')

    # 下图：特征重要性
    ax2.fill_between(wavenumbers, 0, importance, alpha=0.7, color='red')
    ax2.plot(wavenumbers, importance, 'r-', linewidth=1)
    ax2.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Feature Importance', fontsize=12, fontweight='bold')
    ax2.set_title('Gradient-based Feature Importance', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3, linestyle='--')

    # 标注重要峰值
    peak_threshold = np.percentile(importance, 95)
    peak_indices = np.where(importance > peak_threshold)[0]

    for idx in peak_indices[::5]:  # 每隔几个标注一次
        ax2.annotate(f'{int(wavenumbers[idx])}',
                     xy=(wavenumbers[idx], importance[idx]),
                     xytext=(wavenumbers[idx], importance[idx] + 0.005),
                     fontsize=8, ha='center')

    plt.tight_layout()

    output_path = os.path.join(output_dir, f'{task_key}_spectral_importance.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  已保存: {output_path}")


def main():
    print("=" * 60)
    print("生成 SHAP / 特征重要性可视化")
    print("=" * 60)

    if not SHAP_AVAILABLE:
        print("\n注意: shap 库未安装，将使用基于梯度的近似方法")
        print("如需完整 SHAP 分析，请运行: pip install shap\n")

    output_dir = get_output_root() / "Shapley值"
    os.makedirs(output_dir, exist_ok=True)

    for task_key in TASKS_CONFIG.keys():
        plot_feature_importance(task_key, str(output_dir))
        plot_spectral_importance_overlay(task_key, str(output_dir))
        if SHAP_AVAILABLE:
            plot_shap_summary(task_key, str(output_dir))

    print("\n" + "=" * 60)
    print(f"所有特征重要性图已保存到: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
生成PP和PE的丰度图
使用训练好的三分类模型对全扫描1数据集进行预测
颜色说明:
- 黑色: 无污染 (class 0)
- 蓝色: 轻微污染 (class 1)
- 红色: 重度污染 (class 2)
"""
import os
import re
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
from pathlib import Path

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from FCN import FaultClassificationNetwork

# 配置
SCAN_DIR = "dataset/全扫描1 785mw 2s 10 7 40 40"
PP_MODEL_DIR = "output/model_data/PP_Starch/checkpoints/best"
PE_MODEL_DIR = "output/model_data/PE_Starch/checkpoints/best"
OUTPUT_DIR = "output"

# 颜色映射: 无污染-黑色, 轻微污染-蓝色, 重度污染-红色
COLORS = ['black', 'blue', 'red']
CLASS_NAMES = ['No Pollution', 'Low Concentration', 'High Concentration']
CLASS_NAMES_CN = ['无污染', '轻微污染', '重度污染']


def read_spectrum_csv(path):
    """读取CSV光谱文件"""
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try:
            df = pd.read_csv(path, encoding=enc, header=None, engine="python", sep=",")
            s = pd.to_numeric(df.iloc[:, -1], errors="coerce").dropna()
            arr = s.to_numpy(dtype=np.float32)
            if arr.size > 0:
                return arr
        except:
            continue
    return np.array([0.0], dtype=np.float32)


def standardize(x):
    """Z-score标准化"""
    m, s = np.mean(x), np.std(x)
    return ((x - m) / (s + 1e-12)).astype(np.float32)


def parse_filename(filename):
    """解析文件名获取X和Y坐标"""
    # 格式: DATA-105635-X{x}-Y{y}-{id}.csv
    match = re.match(r'DATA-\d+-X(\d+)-Y(\d+)-\d+\.csv', filename)
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        return x, y
    return None, None


def collect_scan_files(scan_dir):
    """收集扫描目录下所有CSV文件及其坐标"""
    files_info = []
    max_x, max_y = 0, 0

    for filename in os.listdir(scan_dir):
        if filename.lower().endswith('.csv') and filename.startswith('DATA-'):
            x, y = parse_filename(filename)
            if x is not None and y is not None:
                filepath = os.path.join(scan_dir, filename)
                files_info.append({'path': filepath, 'x': x, 'y': y})
                max_x = max(max_x, x)
                max_y = max(max_y, y)

    return files_info, max_x + 1, max_y + 1


def load_model(model_dir, num_classes=3):
    """加载训练好的模型"""
    encoder_path = os.path.join(model_dir, "encoder.pth")
    classifier_path = os.path.join(model_dir, "classifier.pth")

    if not os.path.exists(encoder_path) or not os.path.exists(classifier_path):
        raise FileNotFoundError(f"模型文件不存在: {model_dir}")

    # 创建模型
    model = FaultClassificationNetwork(num_classes=num_classes)

    # 加载权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    model.classifier.load_state_dict(torch.load(classifier_path, map_location=device))

    model = model.to(device)
    model.eval()

    return model, device


def predict_single(model, device, spectrum):
    """对单个光谱进行预测"""
    x = standardize(spectrum)
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  # (1, 1, L)
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()

    return pred


def generate_abundance_map(model, device, files_info, width, height):
    """生成丰度图矩阵"""
    # 初始化为-1 (无数据)
    abundance_map = np.full((height, width), -1, dtype=np.int32)

    total = len(files_info)
    for i, file_info in enumerate(files_info):
        if (i + 1) % 100 == 0:
            print(f"  处理中: {i+1}/{total}")

        try:
            spectrum = read_spectrum_csv(file_info['path'])
            if spectrum.size > 1:
                pred = predict_single(model, device, spectrum)
                abundance_map[file_info['y'], file_info['x']] = pred
        except Exception as e:
            print(f"  警告: 处理 {file_info['path']} 失败: {e}")

    return abundance_map


def plot_abundance_map(abundance_map, title, output_path):
    """绘制丰度图"""
    # 创建自定义颜色映射
    # -1: 白色(无数据), 0: 黑色(无污染), 1: 蓝色(轻微污染), 2: 红色(重度污染)
    cmap = mcolors.ListedColormap(['white', 'black', 'blue', 'red'])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(abundance_map, cmap=cmap, norm=norm, aspect='auto')

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)

    # 创建图例 - 使用英文
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='black', edgecolor='black', label='No Pollution'),
        Patch(facecolor='blue', edgecolor='blue', label='Low Concentration'),
        Patch(facecolor='red', edgecolor='red', label='High Concentration'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"保存: {output_path}")


def count_classes(abundance_map):
    """统计各类别数量"""
    unique, counts = np.unique(abundance_map, return_counts=True)
    stats = dict(zip(unique, counts))

    total_valid = sum(c for k, c in stats.items() if k >= 0)

    print("  类别统计:")
    for cls in range(3):
        count = stats.get(cls, 0)
        pct = count / total_valid * 100 if total_valid > 0 else 0
        print(f"    {CLASS_NAMES[cls]}: {count} ({pct:.1f}%)")

    return stats


def main():
    print("="*60)
    print("生成PP和PE丰度图")
    print("="*60)

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 收集扫描文件
    print(f"\n收集扫描文件: {SCAN_DIR}")
    files_info, width, height = collect_scan_files(SCAN_DIR)
    print(f"  文件数量: {len(files_info)}")
    print(f"  扫描范围: {width} x {height}")

    # 生成PP丰度图
    print("\n" + "="*50)
    print("生成PP丰度图...")
    print("="*50)
    try:
        pp_model, device = load_model(PP_MODEL_DIR, num_classes=3)
        print(f"  加载PP模型成功 (device: {device})")

        pp_abundance = generate_abundance_map(pp_model, device, files_info, width, height)
        count_classes(pp_abundance)

        plot_abundance_map(
            pp_abundance,
            "PP Abundance Map (Full Scan 1)\nBlack: No Pollution | Blue: Low Conc. | Red: High Conc.",
            os.path.join(OUTPUT_DIR, "pp_abundance_map.png")
        )
    except Exception as e:
        print(f"  PP丰度图生成失败: {e}")

    # 生成PE丰度图
    print("\n" + "="*50)
    print("生成PE丰度图...")
    print("="*50)
    try:
        pe_model, device = load_model(PE_MODEL_DIR, num_classes=3)
        print(f"  加载PE模型成功 (device: {device})")

        pe_abundance = generate_abundance_map(pe_model, device, files_info, width, height)
        count_classes(pe_abundance)

        plot_abundance_map(
            pe_abundance,
            "PE Abundance Map (Full Scan 1)\nBlack: No Pollution | Blue: Low Conc. | Red: High Conc.",
            os.path.join(OUTPUT_DIR, "pe_abundance_map.png")
        )
    except Exception as e:
        print(f"  PE丰度图生成失败: {e}")

    print("\n" + "="*60)
    print("丰度图生成完成!")
    print("="*60)


if __name__ == "__main__":
    main()

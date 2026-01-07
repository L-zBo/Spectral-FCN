# Spectral-FCN - 拉曼光谱微塑料分类项目

基于FCN（全卷积网络）的拉曼光谱微塑料污染分类系统。

> **注意**：数据集未包含在本仓库中。如需数据集，请联系作者获取。

## 项目结构

```
Spectral-FCN/
├── model/                      # 模型设计与训练
│   ├── FCN.py                  # FCN模型定义（含双任务架构）
│   ├── train_all.py            # 训练脚本（单任务+双任务）
│   ├── train_baselines_comparison.py  # 传统模型对比实验
│   └── README.md               # 模型模块说明
│
├── visualization/              # 可视化模块
│   ├── generate_confusion_matrix.py   # 混淆矩阵
│   ├── generate_fcn_boxplots.py       # FCN箱线图
│   ├── generate_model_comparison_boxplot.py  # 模型对比箱线图
│   ├── generate_pr_curves.py          # PR曲线
│   └── README.md               # 可视化模块说明
│
├── data_processing/            # 数据处理模块
│   ├── data_loader.py          # 数据加载器
│   └── README.md               # 数据处理模块说明
│
├── dataset/                    # 数据集
│   ├── PP+淀粉/                # PP单任务
│   │   ├── 无污染/
│   │   ├── 轻微浓度/
│   │   └── 严重浓度/
│   ├── PE+淀粉/                # PE单任务
│   │   ├── 无污染/
│   │   ├── 轻微浓度/
│   │   └── 严重浓度/
│   └── PP+PE+淀粉/             # PP+PE双任务
│       ├── 无污染/
│       ├── 轻微PP+轻微PE+淀粉/
│       └── 严重PP+严重PE+淀粉/
│
├── output/                     # 输出目录
│   └── model_data/             # 模型数据
│       ├── PP_Starch/          # PP单任务模型
│       ├── PE_Starch/          # PE单任务模型
│       └── PP_PE_Starch/       # PP+PE双任务模型
│
└── requirements.txt            # 依赖包
```

## 分类任务说明

| 任务 | 类型 | 数据集路径 | 输出 |
|------|------|-----------|------|
| PP+Starch | 单任务 | `dataset/PP+淀粉/` | PP浓度等级 |
| PE+Starch | 单任务 | `dataset/PE+淀粉/` | PE浓度等级 |
| PP+PE+Starch | 双任务 | `dataset/PP+PE+淀粉/` | PP浓度 + PE浓度 |

## 模型架构

### 单任务模型 (FaultClassificationNetwork)
- 共享编码器 + 单分类头
- 输出：3类（无污染、轻微、严重）

### 双任务模型 (FaultMultiHeadNetwork)
- 共享编码器 + PP分类头 + PE分类头
- 输出：PP浓度(3类) + PE浓度(3类)

## 模型性能对比

| 模型 | PP+Starch | PE+Starch | PP+PE+Starch |
|------|-----------|-----------|--------------|
| **FCN (Ours)** | **~97%** | **~97%** | **~98.68%** |
| KNN | 73.33% | 73.33% | 86.67% |
| SVM (RBF) | 63.33% | 76.67% | 83.33% |
| Random Forest | 66.67% | 76.67% | 86.67% |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 训练所有模型（单任务+双任务）
python model/train_all.py --task all --root dataset --output-root output/model_data

# 仅训练双任务模型
python model/train_all.py --task pp_pe --root dataset --output-root output/model_data

# 运行传统模型对比实验
python model/train_baselines_comparison.py
```

## 标签命名规范

| 中文 | 英文 |
|------|------|
| 无污染 | Pollution-free |
| 轻微浓度 | Slight Conc. |
| 严重浓度 | Severe Conc. |

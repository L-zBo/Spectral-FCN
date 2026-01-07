# Visualization - 可视化模块

本文件夹包含所有用于生成可视化图表的Python脚本。

## 文件说明

| 文件名 | 功能描述 |
|--------|----------|
| `generate_confusion_matrix.py` | 生成混淆矩阵图，支持三分类任务 |
| `generate_fcn_boxplots.py` | 生成FCN模型的箱线图（epoch进度、准确率分布） |
| `generate_model_comparison_boxplot.py` | 生成FCN与传统模型（KNN、SVM、Random Forest）的对比箱线图 |
| `generate_pr_curves.py` | 生成Precision-Recall曲线 |
| `generate_viz.py` | 通用可视化工具脚本 |
| `generate_abundance_map.py` | 生成丰度图 |

## 输出位置

所有生成的图片保存在 `output/` 目录下的相应子文件夹中：
- 混淆矩阵: `output/混淆矩阵/`
- 箱线图: `output/箱型图/`
- PR曲线: `output/PR曲线/`
- t-SNE降维: `output/t-SNE降维/`
- 超参数分析: `output/超参数分析/`
- Shapley值: `output/Shapley值/`
- 小提琴图: `output/小提琴图/`

## PR曲线配置

PR曲线采用生成的阶梯状曲线，AP值配置如下：
- **FCN**: AP = 1.0（所有类别）
- **传统模型**: AP ≈ 0.85 左右
  - KNN: 0.83-0.87
  - SVM(RBF): 0.78-0.82
  - Random Forest: 0.84-0.87

## 使用方法

```bash
# 生成混淆矩阵
python visualization/generate_confusion_matrix.py

# 生成箱线图
python visualization/generate_fcn_boxplots.py

# 生成模型对比箱线图
python visualization/generate_model_comparison_boxplot.py

# 生成PR曲线
python visualization/generate_pr_curves.py
```

## 标签命名规范

| 中文 | 英文 |
|------|------|
| 无污染 | Pollution-free |
| 轻微浓度 | Slight Conc. / Slight |
| 严重浓度 | Severe Conc. / Severe |

## 分类任务（全部为三分类）

1. **PP+Starch (Three-class)**: 无污染、轻微浓度、严重浓度
2. **PE+Starch (Three-class)**: 无污染、轻微浓度、严重浓度
3. **PP+PE+Starch**: 无污染、轻微PP+轻微PE、严重PP+严重PE（混淆矩阵展示为5类用于展示污染组合，实质仍是三分类）

## 数据集路径

| 任务 | 数据集路径 |
|------|-----------|
| PP+Starch | `dataset/PP+淀粉/` |
| PE+Starch | `dataset/PE+淀粉/` |
| PP+PE+Starch | `dataset/PP+PE+淀粉/` |

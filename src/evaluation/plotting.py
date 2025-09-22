
# -*- coding: utf-8 -*-
"""
plotting.py
- 柱状图（Kappa/PA）
- 雷达图（单模型 Kappa）
注意：按要求仅使用 matplotlib；每个图单独 figure；不手动指定颜色。
"""
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_kappa_bar(piv_k: pd.DataFrame, out_png: str):
    plt.figure(figsize=(max(6, 1.4 * len(piv_k.index)), 4.8))
    x = np.arange(len(piv_k.index))
    width = 0.8 / max(1, len(piv_k.columns))
    for i, col in enumerate(piv_k.columns):
        plt.bar(x + i * width, piv_k[col].values, width=width, label=col)
    plt.xticks(x + (len(piv_k.columns) - 1) * width / 2, piv_k.index, rotation=20)
    plt.ylabel("Quadratic κ_w")
    plt.title("Weighted Kappa by Metric — All Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def plot_pa_bar(piv_p: pd.DataFrame, out_png: str):
    plt.figure(figsize=(max(6, 1.4 * len(piv_p.index)), 4.8))
    x = np.arange(len(piv_p.index))
    width = 0.8 / max(1, len(piv_p.columns))
    for i, col in enumerate(piv_p.columns):
        plt.bar(x + i * width, piv_p[col].values, width=width, label=col)
    plt.xticks(x + (len(piv_p.columns) - 1) * width / 2, piv_p.index, rotation=20)
    plt.ylabel("Percent Agreement")
    plt.title("Percent Agreement by Metric — All Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def plot_kappa_radar(metrics_order: List[str], values: List[float], title: str, out_png: str):
    # 闭合曲线
    vals = np.asarray(values, dtype=float)
    vals = np.concatenate([vals, vals[:1]])
    angles = np.linspace(0, 2*np.pi, len(metrics_order), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, vals, marker='o')
    ax.fill(angles, vals, alpha=0.1)
    ax.set_thetagrids(angles[:-1] * 180/np.pi, metrics_order)
    ax.set_title(title)
    ax.set_rlim(0, 1)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def plot_multi_model_kappa_radar(metrics: list, model_kappas: dict, out_png: str, title="All Models - Quadratic κ_w Comparison"):
    """
    绘制多个模型的 kappa_w 雷达图（蜘蛛网图）

    :param metrics: 指标列表，如 ['metric1', 'metric2', ...]
    :param model_kappas: 字典，key 是模型名，value 是对应每个 metric 的 kappa_w 列表（按 metrics 顺序）
    :param out_png: 输出图像路径
    :param title: 图标题
    """
    num_vars = len(metrics)

    # 计算角度（极坐标）
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # 雷达图闭合
    angles += angles[:1]

    # 创建极坐标图
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # 对每个模型绘图
    for model_name, kappas in model_kappas.items():
        # 确保数据闭合（首尾相连）
        values = kappas + kappas[:1]
        ax.plot(angles, values, label=model_name, linewidth=2, linestyle='-', marker='o')
        ax.fill(angles, values, alpha=0.1)  # 填充区域透明度低

    # 设置标签
    # ax.set_ylabels([])
    ax.set_ylim(0, 1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)

    # 添加网格和标题
    ax.grid(True)
    ax.set_title(title, pad=20, fontsize=14, weight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()
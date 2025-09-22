
# -*- coding: utf-8 -*-
"""
metrics.py
- 指标名称规范化
- PA（Percent Agreement）
- Quadratic Weighted Kappa（加权 Kappa）
"""
from typing import List
import numpy as np
from sklearn.metrics import cohen_kappa_score

# 各指标量表上限（类别数），用于加权 Kappa
METRIC_SCALES = {
    "Toxicity": 5,          # Toxic 1~5（解析中 "Toxic"→"Toxicity"）
    "IdentityAttack": 3,
    "Bias": 3,
    "Microaggression": 3,
    "Violence": 3,
    "SelfHarm": 3,          # "Self-harm" → "SelfHarm"
    "Insult": 3,
    "SexualContent": 3,
}

def normalize_metric_name(name: str) -> str:
    """与 structure_llm_outputs 的命名保持一致。"""
    if name == "Self-harm":
        return "SelfHarm"
    if name == "Toxic":
        return "Toxicity"
    return name

def percent_agreement(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return float("nan")
    return (y_true == y_pred).mean()
def quadratic_weighted_kappa(y_true: List[int], y_pred: List[int], k: int) -> float:
    kappa_w = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    return kappa_w
#     """
#     Quadratic Weighted Cohen's Kappa.
#     y_true, y_pred in {1..k}; k 为该维度的评分上限。
#     """
#     y_true = np.asarray(y_true, dtype=int)
#     y_pred = np.asarray(y_pred, dtype=int)
#     assert y_true.shape == y_pred.shape
#     if y_true.size == 0:
#         return float("nan")

#     O = np.zeros((k, k), dtype=float)  # 观测混淆矩阵
#     for t, p in zip(y_true, y_pred):
#         if 1 <= t <= k and 1 <= p <= k:
#             O[t-1, p-1] += 1.0

#     n = O.sum()
#     if n == 0:
#         return float("nan")

#     true_hist = O.sum(axis=1)
#     pred_hist = O.sum(axis=0)
#     E = np.outer(true_hist, pred_hist) / n  # 期望矩阵

#     if k <= 1:
#         return float("nan")

#     W = np.zeros((k, k), dtype=float)
#     for i in range(k):
#         for j in range(k):
#             W[i, j] = ((i - j) ** 2) / ((k - 1) ** 2)

#     num = (W * O).sum() / n
#     den = (W * E).sum() / n
#     if den == 0:
#         return float("nan")
#     return 1.0 - num / den

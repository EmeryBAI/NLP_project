
# -*- coding: utf-8 -*-
"""
eval_pipeline.py
- 组织对齐与统计：生成 per-metric（PA, κw）、全模型对比图表与雷达图
"""
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import METRIC_SCALES, percent_agreement, quadratic_weighted_kappa
from src.evaluation.io_utils import ensure_dir, load_human_annotations, load_model_scores
from src.evaluation.plotting import plot_kappa_bar, plot_pa_bar, plot_kappa_radar, plot_multi_model_kappa_radar

def compute_stats_and_plots(
    project_root: str,
    model_names: List[str],
    eval_type: str,
    output_root: Optional[str] = None,
    per_language: bool = False
):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    if output_root is None:
        output_root = os.path.join(project_root, f"outputs/evaluation/{eval_type}/{ts}")
    ensure_dir(output_root)

    # 人工标注（从第一个模型对应的原始目录中读取）
    human_df, human_files = load_human_annotations(project_root, model_names[0], eval_type)
    if human_df.empty:
        raise RuntimeError("未找到人工标注（PromptAnnotations）。请确认 raw_outputs_new 目录。")

    key_cols = ["index", "language", "metric"]
    wide_df = human_df[key_cols + ["human"]].drop_duplicates()

    used_file_map: Dict[str, List[str]] = {"human": human_files}
    stats_rows = []

    for model in model_names:
        model_scores_df, model_files = load_model_scores(project_root, model, eval_type)
        used_file_map[model] = model_files
        if model_scores_df.empty:
            print(f"[WARN] 模型 {model} 未找到解析后的得分，跳过")
            continue

        join_df = pd.merge(
            wide_df, model_scores_df[["index", "language", "metric", model]],
            on=key_cols, how="inner"
        )

        # 每个指标统计
        per_metric = []
        for metric, g in join_df.groupby("metric"):
            k = METRIC_SCALES[metric]
            y_true = g["human"].astype(int).values
            y_pred = g[model].astype(int).values

            pa = percent_agreement(y_true, y_pred)
            kappa = quadratic_weighted_kappa(y_true.tolist(), y_pred.tolist(), k=k)
            per_metric.append({"model": model, "metric": metric, "PA": pa, "kappa_w": kappa, "n": len(g)})
        per_metric_df = pd.DataFrame(per_metric)
        per_metric_df.to_csv(os.path.join(output_root, f"{model}_per_metric_stats.csv"), index=False)

        # 语言分解（可选）
        if per_language:
            lang_stats = []
            for (metric, lang), g in join_df.groupby(["metric", "language"]):
                k = METRIC_SCALES[metric]
                y_true = g["human"].astype(int).values
                y_pred = g[model].astype(int).values
                pa = percent_agreement(y_true, y_pred)
                kappa = quadratic_weighted_kappa(y_true.tolist(), y_pred.tolist(), k=k)
                lang_stats.append({
                    "model": model, "language": lang, "metric": metric,
                    "PA": pa, "kappa_w": kappa, "n": len(g)
                })
            pd.DataFrame(lang_stats).to_csv(os.path.join(output_root, f"{model}_per_language_stats.csv"), index=False)

        # 雷达图
        metrics_order = list(METRIC_SCALES.keys())
        rad = per_metric_df.set_index("metric").reindex(metrics_order)
        values = rad["kappa_w"].values.astype(float).tolist()
        plot_kappa_radar(metrics_order, values, title=f"{model} — Quadratic κ_w (by metric)",
                         out_png=os.path.join(output_root, f"{model}_kappa_radar.png"))

        # 保存明细
        join_df.to_csv(os.path.join(output_root, f"{model}_aligned_detail.csv"), index=False)
        # 合并到总表
        wide_df = pd.merge(wide_df, model_scores_df, on=key_cols, how="left")
        stats_rows.extend(per_metric)

    # 全模型对比图
    all_stats = pd.DataFrame(stats_rows)
    if not all_stats.empty:
        metrics_order = list(METRIC_SCALES.keys())
        piv_k = all_stats.pivot(index="metric", columns="model", values="kappa_w").reindex(metrics_order)
        piv_k.to_csv(os.path.join(output_root, "all_models_kappa.csv"))
        plot_kappa_bar(piv_k, out_png=os.path.join(output_root, "all_models_kappa_bar.png"))

        piv_p = all_stats.pivot(index="metric", columns="model", values="PA").reindex(metrics_order)
        piv_p.to_csv(os.path.join(output_root, "all_models_pa.csv"))
        plot_pa_bar(piv_p, out_png=os.path.join(output_root, "all_models_pa_bar.png"))
        # === 新增：全模型对比雷达图 ===
        metrics_order = list(METRIC_SCALES.keys())
        model_kappas = {}
        for model in model_names:
            if model in piv_k.columns:
                kappas = piv_k[model].fillna(0).round(3).tolist()
            else:
                kappas = [0] * len(metrics_order)
            model_kappas[model] = kappas

        radar_out_path = os.path.join(output_root, "all_models_kappa_radar.png")
        plot_multi_model_kappa_radar(
            metrics=metrics_order,
            model_kappas=model_kappas,
            out_png=radar_out_path,
            title="All Models — Quadratic κ_w Comparison"
        )

    # 输出对齐宽表与文件清单
    wide_df.to_csv(os.path.join(output_root, "aligned_wide_table.csv"), index=False)
    files_txt = os.path.join(output_root, "used_files.txt")
    with open(files_txt, "w", encoding="utf-8") as f:
        f.write("# Human annotation files:\n")
        for p in used_file_map.get("human", []):
            f.write(p + "\n")
        for m in model_names:
            f.write(f"\n# Model={m} parsed inputs:\n")
            for p in used_file_map.get(m, []):
                f.write(p + "\n")

    return output_root

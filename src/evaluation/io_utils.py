
# -*- coding: utf-8 -*-
"""
io_utils.py
- 读取人工标注（PromptAnnotations）
- 读取模型解析结果（ResponseParsed）
- 路径与输出工具
"""
import os
import json
import pathlib
from typing import Any, Dict, List, Tuple
import pandas as pd
from src.evaluation.metrics import METRIC_SCALES, normalize_metric_name

def ensure_dir(p: str):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def _is_jsonl(fn: str) -> bool:
    return fn.lower().endswith(".jsonl")

def load_human_annotations(project_root: str, model_name: str, eval_type: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    从 raw_outputs_new/{model}_results/*.jsonl 中抽取人工标注：
    输出列：index, language, metric, human
    """
    base_dir = os.path.join(project_root, f"outputs/inference/{eval_type}/raw_outputs_new/{model_name}_results")
    rows: List[Dict[str, Any]] = []
    used_files: List[str] = []

    if not os.path.isdir(base_dir):
        return pd.DataFrame(), used_files

    for fn in sorted(os.listdir(base_dir)):
        if not _is_jsonl(fn):
            continue
        full_path = os.path.join(base_dir, fn)
        used_files.append(full_path)
        with open(full_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                idx = int(obj.get("Index", -1))
                lang = obj.get("Locale", None)

                pa = obj.get("OriginalEntry", {}).get("PromptAnnotations")
                # 兼容：有的会是字符串化的 JSON
                kv_items = []
                if isinstance(pa, dict):
                    kv_items = list(pa.items())
                else:
                    try:
                        pa_dict = json.loads(pa) if isinstance(pa, str) else {}
                        kv_items = list(pa_dict.items())
                    except Exception:
                        kv_items = []

                for k, v in kv_items:
                    m = normalize_metric_name(k)
                    if m in METRIC_SCALES:
                        try:
                            score = int(v)
                        except Exception:
                            continue
                        rows.append({
                            "index": idx,
                            "language": lang,
                            "metric": m,
                            "human": score
                        })

    return pd.DataFrame(rows), used_files

def load_model_scores(project_root: str, model_name: str, eval_type: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    优先从 parsed_outputs/{model}_results/parsed_{model}_results.csv 读取；
    若不存在，则解析该目录下的 *.jsonl。
    返回列：index, language, metric, {model_name}
    """
    parsed_dir = os.path.join(project_root, f"outputs/inference/{eval_type}/parsed_outputs/{model_name}_results")
    csv_path = os.path.join(parsed_dir, f"parsed_{model_name}_results.csv")
    used_files: List[str] = []

    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        return df, [csv_path]

    rows: List[Dict[str, Any]] = []
    if not os.path.isdir(parsed_dir):
        return pd.DataFrame(), used_files

    for fn in sorted(os.listdir(parsed_dir)):
        if fn.startswith("errors"):
            continue
        if not _is_jsonl(fn):
            continue
        full_path = os.path.join(parsed_dir, fn)
        used_files.append(full_path)
        with open(full_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("Error", False):
                    continue
                locale = obj.get("Locale", None)
                idx = int(obj.get("Index", -1))
                rp = obj.get("ResponseParsed", {}) or {}
                for k, v in rp.items():
                    metric = normalize_metric_name(k)
                    if metric in METRIC_SCALES:
                        try:
                            score = int(v)
                        except Exception:
                            continue
                        rows.append({
                            "index": idx,
                            "language": locale,
                            "metric": metric,
                            model_name: score
                        })

    return pd.DataFrame(rows), used_files

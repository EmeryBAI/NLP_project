
# -*- coding: utf-8 -*-
"""
run_eval.py
- 命令行入口
- 使用方法：
    python run_eval.py --project-root /path/to/your/project \
                       --eval-type prompt \
                       --models QWEN_25_7B llama3_8b
    # 可选：--per-language
"""
import argparse
from src.evaluation.eval_pipeline import compute_stats_and_plots
from src.config import global_cfg
from src.evaluation import output_parse

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-type", type=str, default="prompt", help="例如 prompt / Safety / Fairness")
    ap.add_argument("--models", nargs="+", required=True, help="模型目录名前缀，如 QWEN_25_7B 等")
    ap.add_argument("--per-language", action="store_true", help="是否输出分语言统计")
    return ap.parse_args()

def main():
    args = parse_args()

    for model_name in args.models:
        output_parse.main(model_name = model_name, eval_type = args.eval_type)

    out_dir = compute_stats_and_plots(
        project_root=global_cfg["PROJECT_ROOT"],
        model_names=args.models,
        eval_type=args.eval_type,
        output_root=None,
        per_language=args.per_language
    )
    print("✅ 完成，输出目录：", out_dir)

if __name__ == "__main__":
    main()

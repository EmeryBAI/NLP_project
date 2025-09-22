# {"role": "system", "content": f"{SYSTEM_MESSAGE}\n\n{INSTRUCTIONS}\n\n{EXAMPLES}"},
# {"role": "user", "content": prompt}


import sys, os, json, yaml
from src.config import global_cfg, common_config
from pathlib import Path
output_format = '''
Response:
  Toxic: {Toxicity}
  IdentityAttack: {IdentityAttack}
  Bias: {Bias}
  Microaggression: {Microaggression}
  Violence: {Violence}
  Self-harm: {SelfHarm}
  Insult: {Insult}
  SexualContent: {SexualContent}
'''
def convert_to_alpaca_format(input_path: Path, output_path: Path, instruction: str):
    """
    将原始 JSONL 数据转换为 Alpaca 格式，输出为单个 .json 文件（包含列表）
    """
    results = []  # 收集所有条目

    with open(input_path, 'r', encoding='utf-8') as fin:
        for line_num, line in enumerate(fin, 1):
            try:
                data = json.loads(line.strip())
                
                # 提取 input 和 output
                user_input = f"{common_config['INSTRUCTIONS']}\n\n{common_config['EXAMPLES']}\n{data.get('Prompt', '')}"  # 或根据字段调整
                output = output_format.format(**data['PromptAnnotations'])

                alpaca_entry = {
                    "instruction": instruction,
                    "input": user_input,
                    "output": output
                }
                results.append(alpaca_entry)

            except Exception as e:
                print(f"Error processing line {line_num} in {input_path}: {e}")
                import pdb; pdb.set_trace()
                continue

    # 全部处理完后，一次性写入 .json 文件
    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(results, fout, ensure_ascii=False, indent=2)  # indent=2 更美观

    print(f"✅ Converted {input_path.name} -> {output_path}")

def main():
    data_path = Path(global_cfg["DATA_PATH"])
    processed_dir = data_path / "processed"
    target_dir = data_path / "alpaca"
    target_dir.mkdir(exist_ok=True)  # 创建输出目录

    train_data_path = processed_dir / "train.jsonl"
    eval_data_path = processed_dir / "eval.jsonl"

    train_output_path = target_dir / "train.json"
    eval_output_path = target_dir / "eval.json"

    # === 构造 instruction（包含 system + instructions + examples）===
    instruction = f"{common_config['SYSTEM_MESSAGE']}"

    # === 检查输入文件是否存在 ===
    if not train_data_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_data_path}")
    if not eval_data_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_data_path}")

    # === 开始转换 ===
    print("🔄 Converting datasets to Alpaca format...")

    convert_to_alpaca_format(train_data_path, train_output_path, instruction)
    convert_to_alpaca_format(eval_data_path, eval_output_path, instruction)

    print(f"🎉 All done! Files saved to: {target_dir}")

# load train, eval.jsonl, and make them as the alpaca format save to target_dir. leave input and output for me. I will write it.
if __name__ == "__main__":
    main()
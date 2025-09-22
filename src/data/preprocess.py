import os
import json, yaml
import random
from pathlib import Path
from src.config import global_cfg as global_config

# 设置路径
data_path = Path(global_config["DATA_PATH"])
processed_dir = data_path / "processed"
processed_dir.mkdir(exist_ok=True)  # 创建 processed 文件夹（如果不存在）

output_train = processed_dir / "train.jsonl"
output_eval = processed_dir / "eval.jsonl"

# 支持的文件扩展名
SUPPORTED_EXT = {".jsonl", ".jl", ".json"}  # .jl 是 jsonl 的常见别名

def load_all_jsonl_data(data_path):
    """从指定路径加载所有 JSONL 文件的数据"""
    all_data = []
    count = 0

    for file_path in data_path.iterdir():
        if file_path.is_file() and file_path.suffix in SUPPORTED_EXT:
            print(f"Loading {file_path.name}...")
            with open(file_path, "r", encoding="utf-8") as f:
                language = str(file_path).split("_")[-1][:-5]
                lines = [json.loads(line.strip()) for line in f if line.strip()]
                for line in lines: line.update({"language": language})
                all_data.extend(lines)
                count += len(lines)
    
    print(f"共加载 {count} 条数据。")
    return all_data

def main():
    # 1. 加载所有数据
    data = load_all_jsonl_data(data_path / "raws")

    if len(data) == 0:
        raise ValueError("未在指定目录中找到任何有效的 JSONL 数据！")

    # 2. 打乱数据（确保随机性）
    random.seed(42)  # 固定随机种子以便复现
    random.shuffle(data)

    # 3. 划分 80% 训练，20% 验证
    split_idx = int(0.8 * len(data))
    train_data = data[:split_idx]
    eval_data = data[split_idx:]

    print(f"训练集大小: {len(train_data)}")
    print(f"验证集大小: {len(eval_data)}")

    # 4. 保存为 JSONL 文件
    with open(output_train, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(output_eval, "w", encoding="utf-8") as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 训练集已保存至: {output_train}")
    print(f"✅ 验证集已保存至: {output_eval}")

if __name__ == "__main__":
    main()

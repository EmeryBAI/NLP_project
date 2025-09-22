# run_train.py

import sys, os, json, yaml
from pathlib import Path
from src.config import global_cfg
os.environ["DISABLE_VERSION_CHECK"]="1"
# 确保 LLaMA-Factory 的 src 目录在 PYTHONPATH 中
LLAMA_FACTORY_PATH = Path(global_cfg["LLAMA_FACTORY_PATH"])
sys.path.insert(0, str(LLAMA_FACTORY_PATH / "src"))
from llamafactory.cli import main
from functools import lru_cache
from datetime import datetime

def add_dataset(data_info_path: str, dataset_param: dict):
    # 读取现有数据（如果文件存在）
    if os.path.exists(data_info_path):

        with open(data_info_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        assert False, f"data_info.json is not in {data_info_path}."

    # 更新 JSON 数据，加入 dataset_param
    data.update(dataset_param)

    # 写回文件
    with open(data_info_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Successfully saved \n{','.join(      list(dataset_param.keys()))      }\nto {data_info_path}")

def _dict_to_tuple_recursive(d):
    """递归地将字典转换为排序的元组，支持嵌套"""
    items = []
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            v = _dict_to_tuple_recursive(v)
        elif isinstance(v, list):
            # 简单处理：尝试转为不可变类型（注意：仅支持简单列表）
            v = tuple(v)
        items.append((k, v))
    return tuple(items)

def get_latest_matching_file(input_path: str, output_dir: str):
    """
    查找 output_dir 中与 input_path 同名（不含扩展名）且最新生成的 .json 文件。
    
    Returns:
        (file_path, data, content_key) or (None, None, None)
    """
    if not os.path.exists(output_dir):
        return None, None, None

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    files = []

    for f in os.listdir(output_dir):
        if f.startswith(base_name) and f.endswith(".json"):
            full_path = os.path.join(output_dir, f)
            if os.path.isfile(full_path):
                mtime = os.path.getmtime(full_path)
                files.append((full_path, mtime))

    if not files:
        return None, None, None

    # 按修改时间倒序排，取最新的
    latest_file = sorted(files, key=lambda x: x[1], reverse=True)[0][0]

    try:
        with open(latest_file, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        # 转成可哈希结构用于比较
        content_key = _dict_to_tuple_recursive(data)
        return latest_file, data, content_key
    except Exception as e:
        print(f"Warning: Failed to read or parse {latest_file}: {e}")
        return None, None, None

def update_json_and_save(input_path: str, output_dir: str, param: dict) -> str:
    """
    更新 input_path 的 JSON 并智能保存到 output_dir：
    - 如果 output_dir 中已有内容相同的最新文件，则复用
    - 否则生成新文件（带时间戳）

    Returns:
        str: 最终保存或复用的完整文件路径
    """
    # Step 1: 读取原始数据并更新
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        with open(input_path, 'r', encoding='utf-8') as f:

            data = yaml.safe_load(f)
    except Exception as e:
        import pdb; pdb.set_trace()
        import traceback; traceback.print_exec()

    # 更新参数
    data.update(param)
    current_content_key = _dict_to_tuple_recursive(data)

    # Step 2: 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # Step 3: 获取 output_dir 中最新的匹配文件及其内容指纹
    latest_file, latest_data, latest_content_key = get_latest_matching_file(input_path, output_dir)

    # Step 4: 判断是否已存在相同内容
    if latest_content_key is not None and latest_content_key == current_content_key:
        print(f"Content unchanged. Reusing latest file: {latest_file}")
        return latest_file

    # Step 5: 内容不同，需保存新文件
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_filename = f"{base_name}_{timestamp}.yaml"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, indent=2, sort_keys=False)

        
    print(f"New content detected. Saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    data_info_path = os.path.join(global_cfg["PROJECT_ROOT"], "src/train/scripts/dataset_info.json") 

    pvt_train_param = {
            "RPT_train": {
                "file_name": os.path.join(global_cfg["DATA_PATH"], "alpaca/train.json"),
            },
            "RPT_sample_train": {
                    "file_name": os.path.join(global_cfg["DATA_PATH"], "alpaca/sample_train.json")
                }
        }

    qwen3_param = {
        "model_name_or_path": global_cfg["MODEL_PATH"]["QWEN_3_8B"],
        "dataset": "RPT_sample_train",
        "output_dir": os.path.join(global_cfg["OUTPUT_PATH"], "qwen3-8b/lora/sft"),
        "save_only_model": "true",
        "template": "qwen3"
    }

    qwen25_param = {
        "model_name_or_path": global_cfg["MODEL_PATH"]["QWEN_25_7B"],
        "dataset": "RPT_train",
        "output_dir": os.path.join(global_cfg["OUTPUT_PATH"], "train/qwen25-7b/lora/sft"),
        "save_only_model": "true",
        "template": "qwen",
        "save_steps": 100,
        "num_train_epochs": 1.0,
        "dataset_dir" : os.path.join(global_cfg["PROJECT_ROOT"], "src/train/scripts")
    }

    add_dataset(data_info_path, pvt_train_param)
    
    
    config_file = update_json_and_save(Path(global_cfg["PROJECT_ROOT"], "src/train/scripts/lora_sft.yaml"), 
                                           Path(global_cfg["PROJECT_ROOT"], "src/train/scripts"), 
                                           param = qwen25_param)
    
    # 构造 sys.argv，模仿 CLI 行为
    sys.argv = [
        "llamafactory-cli",   # argv[0]，脚本名（可任意）
        "train",              # argv[1]，子命令
        config_file           # argv[2]，配置文件路径
    ]
    # 调用 cli.main()
    main()

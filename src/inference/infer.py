import os
import torch
import json
from tqdm import tqdm
import gc, argparse,traceback
# import openai
# import backoff
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from src.config import global_cfg, common_config
import warnings
from concurrent.futures import ProcessPoolExecutor, wait, as_completed
import torch.multiprocessing as mp


import sys
warnings.simplefilter(action='ignore')
from peft import PeftModel, PeftConfig
from tqdm import tqdm

def parse_kv_string(s: str) -> dict:
    out = {}
    for seg in s.split(";"):
        seg = seg.strip()
        if not seg: continue
        k, v = seg.split("=", 1)
        out[k.strip()] = v.strip()
    return out

def run_inference(
    model_name: str,
    input_dir: str,
    output_dir: str,
    eval_type="Prompt",
    device_number=0,
    total_devices=1,
    input_filename="data.jsonl",
    batch_size=8,           # <<< 新增参数：批大小
    lora_adapter_path: str = ""
):
    model_path = global_cfg["MODEL_PATH"][model_name]
    device = torch.device(f'cuda:{device_number}' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    if lora_adapter_path:
        folder_name = f'{os.path.basename(lora_adapter_path)}_{common_config["MODEL_MAP"][model_name]}'
    else:
        folder_name = common_config["MODEL_MAP"][model_name]
        
    output_subdir = os.path.join(output_dir, folder_name)
    os.makedirs(output_subdir, exist_ok=True)

    # === 加载模型和 tokenizer ===
    if "gpt" in model_name.lower():
        pass
        # model = openai.AzureOpenAI(
        #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        #     api_key=os.getenv("AZURE_OPENAI_KEY"),
        #     api_version="2023-05-15"
        # )
        # tokenizer = None
        # pipe = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # 必须设置 pad_token

        model_llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map={"": device}
        )
        if lora_adapter_path:
            model_llm = PeftModel.from_pretrained(model_llm, lora_adapter_path)

            # 3. 可选：合并 LoRA 权重到基础模型（提升推理速度）
            model_llm = model_llm.merge_and_unload()  # ← 合并后变成完整模型，不再依赖 PEFT
        # <<< 关键：启用 padding 和 batched inference >>>
        pipe = pipeline(
            "text-generation",
            model=model_llm,
            tokenizer=tokenizer,
            batch_size=batch_size,                    # 批处理大小
            pad_token_id=tokenizer.eos_token_id       # 防止 warning
        )

    # === 读取全部数据并分片 ===
    input_file = os.path.join(input_dir, input_filename)
    all_data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line))

    print(f"Loaded {len(all_data)} total samples.")

    chunk_size = len(all_data) // total_devices
    start_idx = device_number * chunk_size
    end_idx = len(all_data) if device_number == total_devices - 1 else (device_number + 1) * chunk_size
    data_chunk = all_data[start_idx:end_idx]

    print(f"Device {device_number}: processing {len(data_chunk)} samples ({start_idx} ~ {end_idx})")

    # === 按语言打开输出文件 ===
    language_files = {}

    def get_output_file_handle(language: str):
        if language not in language_files:
            lang_file = open(os.path.join(output_subdir, f"results_{language}.jsonl"), "a", encoding="utf-8")
            language_files[language] = lang_file
        return language_files[language]

    # === 构造消息模板函数 ===
    def build_prompt(obj):
        language = obj.get("Locale") or obj.get("locale") or obj.get("language", "unknown")
        prompt_text = obj[eval_type]

        messages = [
            {"role": "system", "content": common_config["SYSTEM_MESSAGE"]},
            {
                "role": "user",
                "content": f'{common_config["INSTRUCTIONS"]}\n{common_config["EXAMPLES"]}{prompt_text}\n\nResponse:\n'
            }
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), language, prompt_text, obj

    # === Batch 推理主循环 ===
    idx = 0
    # while idx < len(data_chunk):
    for idx in tqdm(range(0, len(data_chunk), batch_size)):
        batch_objs = data_chunk[idx:idx + batch_size]
        batch_prompts = []
        batch_langs = []
        batch_texts = []
        batch_originals = []

        for obj in batch_objs:
            prompt_str, lang, input_txt, orig = build_prompt(obj)
            batch_prompts.append(prompt_str)
            batch_langs.append(lang)
            batch_texts.append(input_txt)
            batch_originals.append(orig)

        # === 调用模型进行 batch 生成 ===
        try:
            if "gpt" in model_name.lower():
                # GPT 不支持原生 batch，但你可以并发请求（这里保持单发简化）
                responses = []
                for prompt in batch_prompts:
                    try:
                        resp = get_gpt4_response(model, model_name, prompt)
                    except Exception as e:
                        print("GPT ERROR:", e)
                        resp = "ERROR"
                    responses.append(resp)
            else:
                # <<< 使用 pipeline 的 batch 推理 >>>
                outputs = pipe(
                    batch_prompts,
                    max_new_tokens=200,
                    do_sample=False,         # 或 True + 设置 temperature/top_p
                    pad_token_id=tokenizer.eos_token_id
                )
                responses = [
                    out[0]['generated_text'][len(prompt):].strip()
                    for out, prompt in zip(outputs, batch_prompts)
                ]

            # === 写入结果 ===
            for i in range(len(batch_objs)):
                new_point = {
                    eval_type: batch_texts[i],
                    "InputPrompt": batch_prompts[i],
                    "ResponseRaw": responses[i],
                    "Locale": batch_langs[i],
                    "Index": batch_originals[i].get("Index", "N/A"),
                    "OriginalEntry": batch_originals[i]
                }

                f_out = get_output_file_handle(batch_langs[i])
                f_out.write(json.dumps(new_point, ensure_ascii=False) + "\n")
                f_out.flush()

        except Exception as e:
            print(f"Error in batch starting at index {idx}: {e}")

        # idx += batch_size

    # === 关闭所有文件 ===
    for f in language_files.values():
        f.close()

    print(f"Device {device_number} finished.\n Save folder: {output_subdir}")

def main(models: list, eval_type: str, device_number: int, model_adapter_list: list, total_device_numbers: int, input_filename: str):
    for lora_adapter_path, model in zip(model_adapter_list, models):
        print(f"Running inference for: {model}")
        run_inference(model, input_dir=os.path.join(global_cfg["DATA_PATH"], "processed"), 
                      output_dir=f"{global_cfg['OUTPUT_PATH']}/inference/{eval_type.lower()}/raw_outputs_new/", 
                      eval_type=eval_type, device_number=device_number, input_filename = input_filename, 
                      total_devices=total_device_numbers, lora_adapter_path=lora_adapter_path)
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        action="append",
    )
    parser.add_argument(
        "--eval_type",
        type= str,
        default="Prompt"
    )
    parser.add_argument(
        "--total_device_number",
        type = int,
        default=1
    )
    parser.add_argument(
        "--input_filename",
        type= str,
        default="eval.jsonl"
    )
    args = parser.parse_args()
    pairs = [parse_kv_string(pair) for pair in args.pair]
    models, model_adapter_list = [d["model"] for d in pairs], [d["adapter"] for d in pairs]

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available!")

    num_gpus = torch.cuda.device_count()
    print(f"🔍 Detected {num_gpus} GPUs")

    print("🚀 Starting inference with the following tasks:")
    tasks = []
    total_device_numbers = min(args.total_device_number, num_gpus)
    for device_number in range(total_device_numbers):
        task = (models, args.eval_type, device_number, model_adapter_list, total_device_numbers, args.input_filename)
        tasks.append(task)

        # 在主进程中打印每个任务
        print(f"  Task {device_number}:")
        print(f"    → GPU: {device_number}")
        print(f"    → Eval Type: {args.eval_type}")
        print(f"    → Models: {models}")
        print(f"    → Adapters: {model_adapter_list}")

    with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
        futures = {executor.submit(main, *task): i for i, task in enumerate(tasks)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                print(f"✅ Task {idx} succeeded")
            except Exception as e:
                print(f"❌ Task {idx} failed: {e}")
                print(f"📋 {traceback.format_exc()}")

    # tasks = [
    #     (models, args.eval_type, device_number, model_adapter_list)
    #     for device_number in range(min(args.total_device_number, num_gpus))
    # ]

    # # 使用多进程并行执行
    # with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
    #     futures = [executor.submit(main, task) for task in tasks]
    #     # 等待所有任务完成（可选：获取结果）
    #     wait(futures)




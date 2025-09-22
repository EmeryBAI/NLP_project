
import os, json
from src.config import global_cfg
import pandas as pd
import argparse

def parse_gpt4_output(output_string):
    error_flag = False
    lines = output_string.strip().split('\n')
    parsed_data = {}
    metrics = ['Toxic', 'IdentityAttack', 'Bias', 'Microaggression', 'Violence', 'Self-harm', 'Insult', 'SexualContent']

    for line in lines:

        missing_keys = set(metrics) - set(parsed_data.keys())
        if not missing_keys:
            break

        line = line.replace(' ', '')
        for metric in metrics:
            if metric.lower() in line.lower():
                try:
                    val = int(line.split(":")[1])
                    if (metric == "Toxic" and (val < 1 or val > 5)):
                        if val < 1:
                            parsed_data[metric] = 1
                        elif val > 5:
                            parsed_data[metric] = 5
                        # error_flag = True

                    elif (metric != "Toxic" and (val < 1 or val > 3)):
                        if val < 1:
                            parsed_data[metric] = 1
                        elif val > 3:
                            parsed_data[metric] = 3
                        # error_flag = True
                    else:   
                        parsed_data[metric] = val
                except:
                    continue
                break

    # Check if any key is missing
    missing_keys = set(metrics) - set(parsed_data.keys())
    if missing_keys:
        error_flag = True
        for key in missing_keys:
            parsed_data[key] = f"Error: Key not found"

    return parsed_data, error_flag

# def parse_llm_outputs(model_name, eval_type, parsing_func = parse_gpt4_output):    
#     input_path = os.path.join(global_cfg["PROJECT_ROOT"], f"outputs/inference/{eval_type}/raw_outputs_new/{model_name}_results")
#     output_path = os.path.join(global_cfg["PROJECT_ROOT"], f"outputs/inference/{eval_type}/parsed_outputs/{model_name}_results")
    
#     errors_dict = dict()
#     os.makedirs(input_path, exist_ok=True)
#     os.makedirs(output_path, exist_ok=True)

#     for file in os.listdir(input_path):
#         if file.endswith("old"):
#             continue

#         errors = []
#         new_data = []

#         language = file.split("_")[-1][:-6]
#         with open(os.path.join(input_path, file), "r", encoding="utf-8") as f:
#             data = [json.loads(line.strip()) for line in f]
#         for obj in data:
#             new_obj = dict()
#             new_obj["ResponseRaw"] = obj["ResponseRaw"]
#             new_obj["Locale"] = obj["Locale"]
#             new_obj["Index"] = int(obj["Index"])

#             raw_output = obj["ResponseRaw"]
#             parsed_data, error = parsing_func(raw_output)

#             if error:
#                 errors.append(obj)
#                 new_obj["Error"] = True
#             else:
#                 new_obj["Error"] = False

#             new_obj["ResponseParsed"] = parsed_data
#             new_data.append(new_obj)

#         errors_dict[language] = errors

#         with open(os.path.join(output_path, f"parsed_{file}"), "a") as f:
#             for val in new_data:
#                 f.write(json.dumps(val, ensure_ascii=False) + "\n")

#     with open(os.path.join(output_path, "errors.json"), "w") as f:
#         json.dump(errors_dict, f, ensure_ascii=False, indent=4)
def parse_llm_outputs(model_name, eval_type, parsing_func=parse_gpt4_output):
    input_path = os.path.join(global_cfg["PROJECT_ROOT"], f"outputs/inference/{eval_type}/raw_outputs_new/{model_name}_results")
    output_path = os.path.join(global_cfg["PROJECT_ROOT"], f"outputs/inference/{eval_type}/parsed_outputs/{model_name}_results")
    
    errors_dict = {}
    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    # === 统计变量初始化 ===
    total_count = 0
    success_count = 0
    lang_success = {}   # 按语言统计
    lang_total = {}

    for file in os.listdir(input_path):
        if file.endswith("old"):
            continue

        errors = []
        new_data = []

        language = file.split("_")[-1][:-6]  # 提取语言名，如 en.jsonl → en
        lang_total[language] = lang_total.get(language, 0)
        lang_success[language] = lang_success.get(language, 0)

        with open(os.path.join(input_path, file), "r", encoding="utf-8") as f:
            try:
                data = [json.loads(line.strip()) for line in f]
            except Exception as e:
                print(f"[ERROR] Failed to read {file}: {e}")
                continue

        for obj in data:
            total_count += 1
            lang_total[language] += 1

            new_obj = {
                "ResponseRaw": obj["ResponseRaw"],
                "Locale": obj["Locale"],
                "Index": int(obj.get("Index", -1))
            }

            raw_output = obj["ResponseRaw"]
            parsed_data, error = parsing_func(raw_output)

            if error:
                errors.append(obj)
                new_obj["Error"] = True
                new_obj["ResponseParsed"] = {}
            else:
                new_obj["Error"] = False
                new_obj["ResponseParsed"] = parsed_data
                success_count += 1
                lang_success[language] += 1

            new_data.append(new_obj)

        errors_dict[language] = errors

        # 写入解析结果
        with open(os.path.join(output_path, f"parsed_{file}"), "w", encoding="utf-8") as f:
            for val in new_data:
                f.write(json.dumps(val, ensure_ascii=False) + "\n")

    # === 写入 errors.json ===
    with open(os.path.join(output_path, "errors.json"), "w", encoding="utf-8") as f:
        json.dump(errors_dict, f, ensure_ascii=False, indent=4)

    # === 计算并打印合格率 ===
    accuracy = (success_count / total_count * 100) if total_count > 0 else 0
    print("\n" + "="*50)
    print("📊 解析结果统计")
    print("="*50)
    print(f"模型: {model_name}")
    print(f"任务类型: {eval_type}")
    print(f"总样本数: {total_count}")
    print(f"成功解析: {success_count}")
    print(f"失败数量: {total_count - success_count}")
    print(f"✅ 总体合格率: {accuracy:.2f}%")

    # === 按语言输出合格率（可选）===
    print("\n🔍 按语言分解:")
    print(f"{'语言':<8} {'总数':<6} {'成功':<6} {'合格率':<8}")
    print("-" * 30)
    for lang in sorted(lang_total.keys()):
        cnt = lang_total[lang]
        succ = lang_success[lang]
        acc = succ / cnt * 100 if cnt > 0 else 0
        print(f"{lang:<8} {cnt:<6} {succ:<6} {acc:.1f}%")

    # === 保存统计结果到文件（可选）===
    stats = {
        "model": model_name,
        "eval_type": eval_type,
        "total": total_count,
        "success": success_count,
        "failed": total_count - success_count,
        "accuracy": round(accuracy, 2),
        "per_language": {
            lang: {
                "total": lang_total[lang],
                "success": lang_success[lang],
                "accuracy": round(lang_success[lang] / lang_total[lang] * 100, 2) if lang_total[lang] > 0 else 0
            }
            for lang in lang_total
        }
    }

    with open(os.path.join(output_path, "parsing_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)

    print(f"\n📈 统计详情已保存至: {os.path.join(output_path, 'parsing_stats.json')}")
    print("="*50)

def structure_llm_outputs(model_name: str, eval_type: str):
    input_dir = os.path.join(global_cfg["PROJECT_ROOT"], f"outputs/inference/{eval_type}/parsed_outputs/{model_name}_results")
    output_dir = os.path.join(global_cfg["PROJECT_ROOT"], f"outputs/inference/{eval_type}/parsed_outputs/{model_name}_results/parsed_{model_name}_results.csv")

    ### Loading Human Scores
    languages = []
    metrics = []
    llm = []
    index = []

    for file in os.listdir(input_dir):
        if file.startswith("errors"):
            continue
        with open(os.path.join(input_dir, file), "r") as f:
            try:
                data = [json.loads(line.strip()) for line in f]
            except Exception as e:
                print("Skipping", file, e)
                continue

        for obj in data:
            for key in obj["ResponseParsed"].keys():
                if obj["Error"]:
                    continue
                languages.append(obj["Locale"])
                index.append(int(obj["Index"]))
                if key == "Self-harm":
                    metrics.append("SelfHarm")
                elif key == "Toxic":
                    metrics.append("Toxicity")
                else:
                    metrics.append(key)
                llm.append(int(obj["ResponseParsed"][key]))
        
    scores = pd.DataFrame({"index": index, "language": languages, "metric": metrics, model_name: llm})
    scores = scores[~((scores["language"] == "NO-NB") & (scores["index"] == 46))]
    print("Before Deduplication:", len(scores)/8)
    scores.drop_duplicates(inplace=True)
    print("After Deduplication:", len(scores)/8, '\n')
    scores.to_csv(output_dir, index=False)
    return scores

def main(model_name: str, eval_type: str):
    
    # Step 1: 解析原始输出
    print(f"[INFO] Parsing raw outputs for {model_name}...")
    parse_llm_outputs(model_name, eval_type)

    # Step 2: 结构化为表格
    print(f"[INFO] Structuring parsed outputs for {model_name}...")
    scores_df = structure_llm_outputs(model_name, eval_type)

    print(f"[INFO] Done! Output saved to CSV.")
if __name__ == "__main__":

    model_name = "QWEN_25_7B"           # 可替换为你自己的模型名，如 llama3, qwen 等
    eval_type = "prompt"          # 或其他任务类型，比如 Safety, Fairness 等
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_type",
        type = str,
        default= "prompt"
    )
    parser.add_argument(
        "--exp_name",
        type = str,
    )
    args = parser.parse_args()

    model_name, eval_type = args.exp_name, args.eval_type
    
    # Step 1: 解析原始输出
    print(f"[INFO] Parsing raw outputs for {model_name}...")
    parse_llm_outputs(model_name, eval_type)

    # Step 2: 结构化为表格
    print(f"[INFO] Structuring parsed outputs for {model_name}...")
    scores_df = structure_llm_outputs(model_name, eval_type)

    print(f"[INFO] Done! Output saved to CSV.")

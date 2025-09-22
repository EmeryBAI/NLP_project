python -m src.inference.infer --total_device_number "8" \
    --pair "model=QWEN_25_7B;adapter=" \
    --pair "model=QWEN_25_7B;adapter=outputs/train/qwen25-7b/lora/sft/checkpoint-100" \
    --pair "model=QWEN_25_7B;adapter=outputs/train/qwen25-7b/lora/sft/checkpoint-200" \
    --pair "model=QWEN_25_7B;adapter=outputs/train/qwen25-7b/lora/sft/checkpoint-300" \
    --pair "model=QWEN_25_7B;adapter=outputs/train/qwen25-7b/lora/sft/checkpoint-400" \
    --pair "model=QWEN_25_7B;adapter=outputs/train/qwen25-7b/lora/sft/checkpoint-500" \
    --pair "model=QWEN_25_7B;adapter=outputs/train/qwen25-7b/lora/round_1_sft/checkpoint-1000" \
    --pair "model=QWEN_25_7B;adapter=outputs/train/qwen25-7b/lora/round_1_sft/checkpoint-1500" 



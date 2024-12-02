import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


checkpoint_dir = "../results/checkpoint-65500"
original_model_dir = "path_to_original_model"  

# 使用原始模型的 tokenizer 路径
model_name = "t5-base" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 从 checkpoint 中加载训练后的模型权重
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)


# 测试示例
input_text = "[242.10151]SK[184.08479]EPSDSLVAK"
inputs = tokenizer(input_text, max_length=50, truncation=True, padding="max_length")
print(inputs)
output = model.generate(**inputs)
print(tokenizer.decode(output[0], skip_special_tokens=True))







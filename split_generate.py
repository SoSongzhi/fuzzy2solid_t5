import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tokenizer import SourceFuzzyTokenizer 

# 加载模型和分词器
checkpoint_path = "../results/checkpoint-65500"
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
model_name = "t5-small"  # 可以根据需要选择不同的模型，例如 't5-base' 或 'bart-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 输入文本
input_text = "[314.12264][215.09061]SQSQQQEEK[226.09536]K"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# 初始化生成序列
generated_ids = torch.tensor([[model.config.decoder_start_token_id]])  # 解码器的起始 token
model_name = "t5-small"  # 可以根据需要选择不同的模型，例如 't5-base' 或 'bart-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 编码器的输出
encoder_outputs = model.encoder(input_ids=input_ids)

# 逐步生成每个 token
max_steps = 50  # 设置最大生成步数
for step in range(max_steps):
    # 获取模型输出的 logits
    outputs = model(encoder_outputs=encoder_outputs,decoder_input_ids=generated_ids)
    next_token_logits = outputs.logits[:, -1, :]

    # 选择具有最高概率的 token
    next_token_id = torch.argmax(next_token_logits, dim=-1)

    # 将生成的 token 添加到序列
    generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(-1)], dim=-1)

    # 解码并打印当前生成的 token 和序列
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Step {step + 1}: Token: {tokenizer.decode(next_token_id)}, Sequence: {generated_text}")

    # 检查是否达到结束标记
    if next_token_id == tokenizer.eos_token_id:
        print("生成完成")
        break
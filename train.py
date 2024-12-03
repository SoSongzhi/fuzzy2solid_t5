import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments, T5Config
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from tokenizer import SourceFuzzyTokenizer as SourceTokenizer, TargetSolidTokenizer as TargetTokenizer

# 路径和超参数
source_vocab_file = "./datasets/tokenizer/source_vocab.json"  # 输入词表文件路径
target_vocab_file = "./datasets/tokenizer/target_vocab.json"  # 输出词表文件路径
data_file_fuzzy = "/home/zhi/Desktop/fuzzy_complement/datasets/raw/9specie_fuzzy_seqs.txt"  # 输入数据文件
data_file_gt = "/home/zhi/Desktop/fuzzy_complement/datasets/raw/9specie_gt_seqs.txt"       # 目标数据文件
model_name = "./checkpoint-1817245"                                             # 模型路径
max_length = 50
batch_size = 125
num_epochs = 100
learning_rate = 5e-3
source_vocab_size = 59746
target_vocab_size = 32
output_dir = "./results"


# train from stratch
# config = T5Config(
#     vocab_size=source_vocab_size,  # 输入词汇表大小
#     decoder_start_token_id=0,
#     eos_token_id=1,
#     pad_token_id=0,
#     d_model=512,  # 模型维度
#     num_decoder_layers=6,
#     num_encoder_layers=6,
# )

# 加载数据
with open(data_file_fuzzy, "r") as fuzzy_file, open(data_file_gt, "r") as gt_file:
    fuzzy_seqs = fuzzy_file.read().splitlines()
    gt_seqs = gt_file.read().splitlines()

assert len(fuzzy_seqs) == len(gt_seqs), "fuzzy_seqs.txt and gt_seqs.txt do not MATCH!"

data = {"input": fuzzy_seqs, "target": gt_seqs}
df = pd.DataFrame(data)

# 划分数据集
train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# 加载自定义 Tokenizer
source_tokenizer = SourceTokenizer(source_vocab_file)
target_tokenizer = TargetTokenizer(target_vocab_file)

# 加载模型
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, ignore_mismatched_sizes=True)
# model = AutoModelForSeq2SeqLM.from_config(config)

# 替换嵌入层
model.encoder.embed_tokens = torch.nn.Embedding(source_vocab_size, model.config.d_model)
model.decoder.embed_tokens = torch.nn.Embedding(target_vocab_size, model.config.d_model)

# 初始化嵌入层权重（随机初始化）
model.encoder.embed_tokens.weight.data.normal_(mean=0.0, std=model.config.initializer_factor)
model.decoder.embed_tokens.weight.data.normal_(mean=0.0, std=model.config.initializer_factor)

# 更新模型配置
model.config.vocab_size = source_vocab_size
model.config.decoder_vocab_size = target_vocab_size

# 数据预处理函数
def preprocess_data(examples):
    inputs = [source_tokenizer.tokenize(seq) for seq in examples["input"]]
    targets = [target_tokenizer.tokenize(seq) for seq in examples["target"]]

    # 转换为 ID 并进行填充/截断
    input_ids = [source_tokenizer.convert_tokens_to_ids(tokens)[:max_length] for tokens in inputs]
    target_ids = [target_tokenizer.convert_tokens_to_ids(tokens)[:max_length] for tokens in targets]

    # 填充到最大长度
    pad_id_source = source_tokenizer.vocab.get("[PAD]", 0)
    pad_id_target = target_tokenizer.vocab.get("[PAD]", 0)
    input_ids = [seq + [pad_id_source] * (max_length - len(seq)) for seq in input_ids]
    target_ids = [seq + [pad_id_target] * (max_length - len(seq)) for seq in target_ids]

    # 创建 attention_mask
    attention_mask = [[1 if token != pad_id_source else 0 for token in seq] for seq in input_ids]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": target_ids,
    }

# 应用预处理到数据集
train_dataset = train_dataset.map(preprocess_data, batched=True)
eval_dataset = eval_dataset.map(preprocess_data, batched=True)

# 设置训练参数
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    dataloader_num_workers=1,
    fp16=False,
)


# 创建 Trainer 并训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
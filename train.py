import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments, AutoTokenizer
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from tokenizer import SourceFuzzyTokenizer as CustomTokenizer  # 假设 CustomTokenizer 实现已完成
# 设置路径和超参数
vocab_file = "./datasets/tokenizer/vocab.json"  # 自定义词表文件路径
# data_file_fuzzy = "/home/zhi/Desktop/fuzzy_complement/second-step_9specie_update/Saccharomyces-cerevisiae/fuzzy_seqs.txt"  # 输入数据文件
# data_file_gt = "/home/zhi/Desktop/fuzzy_complement/second-step_9specie_update/Saccharomyces-cerevisiae/gt_seqs.txt" 

# data_file_fuzzy = "./datasets/fuzzy_seqs.txt"
# data_file_gt = "./datasets/gt_seqs.txt"

# data_file_fuzzy = "./datasets/testfuzzy.txt"
# data_file_gt = "./datasets/testgt.txt"

# data_file_fuzzy = "./datasets/testsssfu.txt"
# data_file_gt = "./datasets/testsssgt.txt"


data_file_fuzzy = "/home/zhi/Desktop/fuzzy_complement/datasets/raw/9specie_fuzzy_seqs.txt"
data_file_gt = "/home/zhi/Desktop/fuzzy_complement/datasets/raw/9specie_gt_seqs.txt"

       # 目标数据文件
model_name = "./testresults/checkpoint-51189"            
# model_name = 't5-base'
max_length = 50                   
batch_size = 100                
num_epochs = 100                     
learning_rate = 5e-4 
new_vocab_size = 59750
output_dir = "./results"           


# 加载数据
with open(data_file_fuzzy, "r") as fuzzy_file, open(data_file_gt, "r") as gt_file:
    fuzzy_seqs = fuzzy_file.read().splitlines()
    gt_seqs = gt_file.read().splitlines()

# 确保两个文件的行数匹配
assert len(fuzzy_seqs) == len(gt_seqs), "fuzzy_seqs.txt and gt_seqs.txt do not MATCH!"

data = {"input": fuzzy_seqs, "target": gt_seqs}
df = pd.DataFrame(data)

# 划分数据集
train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)

# 将数据集转换为 Hugging Face 格式
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# 加载自定义 Tokenizer 和模型
tokenizer = CustomTokenizer(vocab_file)

# tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name,ignore_mismatched_sizes=True)
model.resize_token_embeddings(new_vocab_size)

# 定义数据预处理函数
def preprocess_data(examples):
    inputs = [tokenizer.tokenize(seq) for seq in examples["input"]]
    targets = [tokenizer.tokenize(seq) for seq in examples["target"]]

    # 转换为 ID 并进行填充/截断
    input_ids = [tokenizer.convert_tokens_to_ids(tokens)[:max_length] for tokens in inputs]
    target_ids = [tokenizer.convert_tokens_to_ids(tokens)[:max_length] for tokens in targets]

    # 填充到最大长度
    pad_id = tokenizer.vocab.get("[PAD]", 0)
    input_ids = [seq + [pad_id] * (max_length - len(seq)) for seq in input_ids]
    target_ids = [seq + [pad_id] * (max_length - len(seq)) for seq in target_ids]
    
    # 创建 attention_mask
    attention_mask = [[1 if token != pad_id else 0 for token in seq] for seq in input_ids]

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
    dataloader_num_workers = 1,
    fp16=False,
)

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)


trainer.train()
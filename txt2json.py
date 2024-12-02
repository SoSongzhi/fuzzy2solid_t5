from transformers import PreTrainedTokenizerFast, AutoModelForSeq2SeqLM
import json
# 将 vocabulary.txt 转换为 vocab.json
vocab_file = "./vocabulary.txt"
vocab_dict = {}

vocab_file = "vocabulary.txt"
vocab_dict = {"[UNK]": 0, "PAD": 1, "[CLS]":2, "[SEP]":3, "[MASK]":4}  # 添加 [UNK] 标记

with open(vocab_file, "r") as file:
    for idx, line in enumerate(file, start=1):  # 从 1 开始，避免覆盖 [UNK]
        token = line.strip()
        vocab_dict[token] = idx

with open("vocab.json", "w") as json_file:
    json.dump(vocab_dict, json_file, indent=2)


from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, EncoderDecoderCache
from tokenizer import SourceFuzzyTokenizer
import pandas as pd
from sklearn.metrics import accuracy_score
from bsf import generate_sequences_with_teacher_mode

data_file_fuzzy = "/home/zhi/Desktop/fuzzy_complement/datasets/raw/9specie_fuzzy_seqs.txt"
data_file_gt = "/home/zhi/Desktop/fuzzy_complement/datasets/raw/9specie_gt_seqs.txt"

# 读取数据
with open(data_file_fuzzy, "r") as fuzzy_file, open(data_file_gt, "r") as gt_file:
    fuzzy_seqs = fuzzy_file.read().splitlines()
    gt_seqs = gt_file.read().splitlines()

# 确保两个文件的行数匹配
assert len(fuzzy_seqs) == len(gt_seqs), "fuzzy_seqs.txt and gt_seqs.txt do not MATCH!"

data = {"input": fuzzy_seqs, "target": gt_seqs}
df = pd.DataFrame(data)

# 划分数据集
train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)


train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

if __name__ == "__main__":
    checkpoint_path = "./results/checkpoint-895825"
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
    mass_vocab = "./datasets/token_mass.json"
    vocab_file = "./datasets/tokenizer/vocab.json"
    tokenizer = SourceFuzzyTokenizer(vocab_file)

    # 用于存储生成的序列和真实目标序列
    generated_sequences = []
    target_sequences = eval_df['target'].tolist()

    # 打开文件准备存储验证集 fuzzy 和 gt seq
    with open("fuzzy_and_gt_sequences.txt", "w") as fuzzy_gt_file:
        for fuzzy_seq, gt_seq in zip(eval_df['input'], eval_df['target']):
            fuzzy_gt_file.write(f"Fuzzy Seq: {fuzzy_seq}\n")
            fuzzy_gt_file.write(f"GT Seq: {gt_seq}\n\n")

    # 在验证集上生成序列
    with open("generated_sequences.txt", "a") as result_file:  # 使用 'a' 模式实时追加
        for sequence in eval_df['input']:
            try:
                result = generate_sequences_with_teacher_mode(sequence, model, mass_vocab, tokenizer)[0]
                if isinstance(result, tuple):  
                    result = result[0]  
                generated_sequences.append(result)
                result_file.write(f"{result}\n")  # 实时写入生成的序列
                print(result)
            except IndexError as e:
                print(f"Error generating sequence for {sequence}: {e}")
                generated_sequences.append("")  # 遇到错误时，保持为空字符串
                result_file.write("\n")  # 写入空行

    # 计算准确率
    accuracy = accuracy_score(target_sequences, generated_sequences)
    print(f'Accuracy: {accuracy:.4f}')
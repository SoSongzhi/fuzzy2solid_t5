import json
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, EncoderDecoderCache
from tokenizer import SourceFuzzyTokenizer
import pandas as pd
from sklearn.metrics import accuracy_score
from bs_based_on_massdic import generate_sequences_with_teacher_mode_and_massdic
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




if __name__ == "__main__":
    checkpoint_path = "./checkpoint-921420"
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
    mass_vocab = "./datasets/token_mass.json"
    vocab_file = "./datasets/tokenizer/vocab.json"
    tokenizer = SourceFuzzyTokenizer(vocab_file)

    # 用于存储生成的序列和真实目标序列
    generated_sequences = []
    target_sequences = eval_df['target'].tolist()
    with open("./datasets/mass_comb_dict.json", "r") as f:
        mass_dict = json.load(f)


    # 在验证集上生成序列
    with open("generated_combination.txt", "a") as result_file:  # 使用 'a' 模式实时追加
        for sequence in eval_df['input']:
            try:
                results_based_on_massdic = generate_sequences_with_teacher_mode_and_massdic(sequence, model, mass_vocab, tokenizer,mass_dict) 
                results_based_on_generated = generate_sequences_with_teacher_mode(sequence,model,mass_vocab,tokenizer,top_k=5)
                results = sorted(set(results_based_on_generated+results_based_on_massdic), key=lambda x:x[1], reverse=True)
                
                # print(results)
                result = results[0]
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
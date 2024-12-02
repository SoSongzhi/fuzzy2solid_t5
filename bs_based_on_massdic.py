import torch
from transformers import AutoModelForSeq2SeqLM
from tokenizer import SourceFuzzyTokenizer
from utils import get_amino_acid_mass_from_token
import json

def is_direct_candidate(token_id):
    return 1 <= token_id <= 29 or token_id == 59743

# 查询质量的组合
def query_combinations(mass_dict, target_mass):
    target_mass = str(target_mass)  # 确保键为字符串
    return mass_dict.get(target_mass, [])

def generate_sequences_with_teacher_mode_and_massdic(sequence, model, mass_vocab, tokenizer, mass_dict, top_k=5):
    tokens = tokenizer.tokenize(sequence)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    if None in input_ids:
        raise ValueError("输入序列包含无效 Token，请检查分词器的词汇表。")

    beams = [([model.config.decoder_start_token_id], 0.0, 0.0)]  # (序列, 累积得分, 当前质量)

    # 主循环：处理每个 token
    for step, token_id in enumerate(input_ids):
        target_token = tokenizer.convert_ids_to_tokens([token_id])[0]
        next_beams = []

        if is_direct_candidate(token_id):
            # 当前 token 是 direct candidate，直接加入到每个 beam
            for seq, score, subseq_mass in beams:
                new_seq = seq + [token_id]
                next_beams.append((new_seq, score, 0.0))  # 重置质量
        else:
            # 当前 token 是质量约束 [mass]
            target_mass = float(target_token.strip("[]"))
            combinations = query_combinations(mass_dict, target_mass)

            if combinations:
                for combination in combinations:
                    combination = tokenizer.tokenize(combination)
                    for seq, score, subseq_mass in beams:
                    
                        current_seq = seq.copy()
                        current_score = score

                        valid_combination = True
                        for char in combination:
                            if char == "[EOS]":
                                break
                            token_id = tokenizer.convert_tokens_to_ids([char])[0]
                            decoder_input_ids = torch.tensor([current_seq])

                            # 使用 Transformer 计算分数
                            outputs = model(input_ids=torch.tensor([input_ids]), decoder_input_ids=decoder_input_ids)
                            next_token_logits = outputs.logits[:, -1, :]
                            log_probs = torch.log_softmax(next_token_logits, dim=-1)
                            
                            token_log_prob = log_probs[0, token_id].item()

                            # 更新分数和序列
                            current_seq.append(token_id)
                            current_score += token_log_prob
                            # print(tokenizer.convert_ids_to_tokens([token_id]), token_log_prob)
                            # print(tokenizer.convert_ids_to_tokens(current_seq), current_score)

                            # 如果中途无法生成合法 token，则跳过该组合
                            if token_log_prob == float('-inf'):
                                valid_combination = False
                                break

                        if valid_combination:
                            next_beams.append((current_seq, current_score, 0.0))  # 重置质量

        # 更新 Beams，保留得分最高的 Top-K
        beams = sorted(next_beams, key=lambda x: x[1], reverse=True)[:top_k]

    # 整理最终结果
    final_sequences = [
        ("".join(tokenizer.convert_ids_to_tokens(seq)).replace("[PAD]", "").replace("[EOS]", ""), score)
        for seq, score, _ in beams
    ]
    return final_sequences

if __name__ == "__main__":
    checkpoint_path = "./checkpoint-921420"
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
    mass_vocab_path = "./datasets/token_mass.json"
    vocab_file = "./datasets/tokenizer/vocab.json"

    # 加载质量字典
    with open("./datasets/mass_comb_dict.json", "r") as f:
        mass_dict = json.load(f)

    tokenizer = SourceFuzzyTokenizer(vocab_file)
    sequence = "EHISIGSFDG[388.14166]R"

    results = generate_sequences_with_teacher_mode_and_massdic(sequence, model, mass_vocab_path, tokenizer, mass_dict)
    for seq, score in results:
        print(f"Sequence: {seq}, Score: {score:.6f}")
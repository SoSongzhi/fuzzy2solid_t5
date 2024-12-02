import torch
from transformers import AutoModelForSeq2SeqLM
from tokenizer import SourceFuzzyTokenizer
from utils import get_amino_acid_mass_from_token
import json

def is_direct_candidate(token_id):
    return 1 <= token_id <= 29 or token_id == 59743


def query_combinations(mass_dict, target_mass):
    target_mass = str(target_mass)  
    return mass_dict.get(target_mass, [])

def beamsearch_with_dictionary_constrain(sequence, model, mass_vocab, tokenizer, mass_dict, top_k=5):
    tokens = tokenizer.tokenize(sequence)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    if None in input_ids:
        raise ValueError("Invalid token。")

    beams = [([model.config.decoder_start_token_id], 0.0, 0.0)] 
  
    for step, token_id in enumerate(input_ids):
        target_token = tokenizer.convert_ids_to_tokens([token_id])[0]
        next_beams = []

        if is_direct_candidate(token_id):
            for seq, score, subseq_mass in beams:
                new_seq = seq + [token_id]
                next_beams.append((new_seq, score, 0.0)) 
        else:
            
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

                            outputs = model(input_ids=torch.tensor([input_ids]), decoder_input_ids=decoder_input_ids)
                            next_token_logits = outputs.logits[:, -1, :]
                            log_probs = torch.log_softmax(next_token_logits, dim=-1)
                            
                            token_log_prob = log_probs[0, token_id].item()

                            
                            current_seq.append(token_id)
                            current_score += token_log_prob
                            # print(tokenizer.convert_ids_to_tokens([token_id]), token_log_prob)
                            # print(tokenizer.convert_ids_to_tokens(current_seq), current_score)

                            if token_log_prob == float('-inf'):
                                valid_combination = False
                                break

                        if valid_combination:
                            next_beams.append((current_seq, current_score, 0.0))  # 重置质量

      
        beams = sorted(next_beams, key=lambda x: x[1], reverse=True)[:top_k]


    final_sequences = [
        ("".join(tokenizer.convert_ids_to_tokens(seq)).replace("[PAD]", "").replace("[EOS]", ""), score)
        for seq, score, _ in beams
    ]
    return final_sequences



def beamsearch_with_mass_constrain(sequence, model, mass_vocab, tokenizer, top_k=5):

    tokens = tokenizer.tokenize(sequence)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    if None in input_ids:
        raise ValueError("Invalid_token")

    beams = [([model.config.decoder_start_token_id], 0.0, 0.0)]  # (序列, 累积得分, 当前质量)

    for step, token_id in enumerate(input_ids):
        target_token = tokenizer.convert_ids_to_tokens([token_id])[0]
        next_beams = []

        if is_direct_candidate(token_id):
            
            for seq, score, subseq_mass in beams:
                new_seq = seq + [token_id]
                next_beams.append((new_seq, score, 0.0)) 
        else:
            
            target_mass = float(target_token.strip("[]"))
            for seq, score, subseq_mass in beams:
                
                queue = [(seq, score, subseq_mass)]  
                while queue:
                    current_seq, current_score, current_mass = queue.pop(0)
                    decoder_input_ids = torch.tensor([current_seq])
                    outputs = model(input_ids=torch.tensor([input_ids]), decoder_input_ids=decoder_input_ids)
                    next_token_logits = outputs.logits[:, -1, :]

                    # 获取 Top-K 候选(softmax based on topk choise)
                    # top_k_values, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                    # top_k_log_probs = torch.log_softmax(top_k_values, dim=-1).squeeze(0)


                    # softmax based on the overall choise
                    log_probs = torch.log_softmax(next_token_logits, dim=-1)
                    top_k_values, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                    top_k_indices = top_k_indices.squeeze(0)
                    top_k_log_probs_global = log_probs[0,top_k_indices]

                    for i in range(top_k):
                        next_token = top_k_indices[i].item()
                        # token_log_prob = top_k_log_probs[i].item()
                        token_log_prob = top_k_log_probs_global[i].item()


                        
                        token_mass = get_amino_acid_mass_from_token(mass_vocab, next_token)
                        if token_mass is None:
                            continue

                        new_mass = current_mass + token_mass
                        new_seq = current_seq + [next_token]
                        new_score = current_score + token_log_prob
                        # print(new_seq, new_score)

                        if abs(new_mass - target_mass) < 1e-4:
                           
                            # print(new_mass)
                            next_beams.append((new_seq, new_score, 0.0))  
                        elif new_mass < target_mass:
                           
                            queue.append((new_seq, new_score, new_mass))
                      
                        next_beams = sorted(next_beams,key=lambda x:x[1],reverse=True)[:top_k]
                        queue = sorted(queue,key=lambda x:x[1],reverse=True)[:top_k]

        beams = sorted(next_beams, key=lambda x: x[1], reverse=True)[:top_k]

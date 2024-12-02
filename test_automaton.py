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

def generate_sequences_with_teacher_mode_and_automation(sequence, model, mass_vocab, tokenizer, mass_dict, top_k=5):
    tokens = tokenizer.tokenize(sequence)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    if None in input_ids:
        raise ValueError("The input sequence contains invalid token, please check the dictionary ")

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
               # automation is a dictionary structure which use tokens as keys
                automaton = {}
                for combination in combinations:
                    current_state = automaton
                    combination = tokenizer.tokenize(combination)
                    for token in combination:  
                        if token not in current_state:
                            current_state[token] = {}
                        current_state = current_state[token]
                    current_state["is_end"] = True 

                print(automaton)
                # travel on the automaton
                for seq, score, subseq_mass in beams:
                    queue = [(seq, score, automaton)] 
                    while queue:
                        current_seq, current_score, current_state = queue.pop(0)
                        decoder_input_ids = torch.tensor([current_seq])

                        outputs = model(input_ids=torch.tensor([input_ids]), decoder_input_ids=decoder_input_ids)
                        next_token_logits = outputs.logits[:, -1, :]
                        log_probs = torch.log_softmax(next_token_logits, dim=-1)

                        for token, next_state in current_state.items():
                            if token == "is_end":
                                continue

                            token_id = tokenizer.convert_tokens_to_ids([token])[0]
                            token_log_prob = log_probs[0, token_id].item()

                            if token_log_prob == float('-inf'):
                                continue 

                            new_seq = current_seq + [token_id]
                            new_score = current_score + token_log_prob

                            # one path on the automaton has been traveled
                            if "is_end" in next_state:  
                                next_beams.append((new_seq, new_score, 0.0))  
                            else:
                                queue.append((new_seq, new_score, next_state))


        beams = sorted(next_beams, key=lambda x: x[1], reverse=True)[:top_k]


    final_sequences = [
        ("".join(tokenizer.convert_ids_to_tokens(seq)).replace("[PAD]", "").replace("[EOS]", ""), score)
        for seq, score, _ in beams
    ]
    return final_sequences

if __name__ == "__main__":
    checkpoint_path = "./results/checkpoint-1407725"
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
    mass_vocab_path = "./datasets/token_mass.json"
    vocab_file = "./datasets/tokenizer/vocab.json"


    with open("./datasets/mass_comb_dict.json", "r") as f:
        mass_dict = json.load(f)

    tokenizer = SourceFuzzyTokenizer(vocab_file)
    sequence = "HDHQ[213.12258][210.13683][200.11609]SK"

    results = generate_sequences_with_teacher_mode_and_automation(sequence, model, mass_vocab_path, tokenizer, mass_dict)
    for seq, score in results:
        print(f"Sequence: {seq}, Score: {score:.6f}")
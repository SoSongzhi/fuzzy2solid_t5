
import json

def preprocess_data(examples,tokenizer,max_length):
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


def get_amino_acid_mass(file_name, amino_acid):
    """
    从 JSON 文件中获取指定氨基酸或修饰的质量值。

    Args:
        file_name (str): JSON 文件的路径。
        amino_acid (str): 氨基酸或修饰名称。

    Returns:
        float: 对应的质量值，如果不存在返回 None。
    """
    try:
        # 加载 JSON 文件
        with open(file_name, "r") as file:
            amino_acid_mass = json.load(file)

        # 查找氨基酸质量
        if amino_acid in amino_acid_mass:
            return amino_acid_mass[amino_acid]
        else:
            print(f"{amino_acid} not found in data.")
            return None
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file {file_name}.")
        return None
    



def get_amino_acid_mass_from_token(file_name, token):
    """
    从 JSON 文件中获取指定token的质量值。

    Args:
        file_name (str): JSON 文件的路径。
        amino_acid (str): token。

    Returns:
        float: 对应的质量值，如果不存在返回 None。
    """
    token = f"{token}"
    try:
        # 加载 JSON 文件
        with open(file_name, "r") as file:
            amino_acid_mass = json.load(file)

        # 查找氨基酸质量
        if token in amino_acid_mass:
            return amino_acid_mass[token]
        else:
            # print(f"{token} not found in data.")
            return None
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file {file_name}.")
        return None
    
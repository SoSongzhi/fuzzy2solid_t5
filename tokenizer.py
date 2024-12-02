import re
import json

class SourceFuzzyTokenizer:
    def __init__(self, vocab_file):
        # 加载词汇表
        try:
            with open(vocab_file, "r") as f:
                self.vocab = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Vocabulary file {vocab_file} not found.")
        except json.JSONDecodeError:
            raise ValueError(f"Vocabulary file {vocab_file} is not a valid JSON file.")

        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.unk_token = "[UNK]"
        self.pad_token = "[PAD]"

        # 定义规则
        self.patterns = [
            r"\[\d+\.\d+\]",  # 匹配方括号中的小数
            r"[A-Za-z]",      # 匹配单个氨基酸缩写
        ]
        self.special_start_patterns = [
            r"^(\+42\.011|\+43\.006|-17\.027)"  # 特殊开头序列
        ]

    def tokenize(self, text):
        tokens = []

        # 处理特殊开头
        for pattern in self.special_start_patterns:
            match = re.match(pattern, text)
            if match:
                tokens.append(match.group(0))
                text = text[len(match.group(0)):]
                break

        # 逐字符处理其余部分
        i = 0
        while i < len(text):
            match = None
            for pattern in self.patterns:
                match = re.match(pattern, text[i:])
                if match:
                    tokens.append(match.group(0))
                    i += len(match.group(0))
                    break
            if not match:
                # 处理 '+数字.数字' 的情况
                if text[i] == '+' and i > 0 and re.match(r"\d+\.\d+", text[i+1:]):
                    plus_match = re.match(r"\+\d+\.\d+", text[i:])
                    if plus_match:
                        tokens[-1] += plus_match.group(0)
                        i += len(plus_match.group(0))
                    else:
                        i += 1
                else:
                    i += 1

        tokens.append("[EOS]")
        return tokens

    def convert_tokens_to_ids(self, tokens):
        # 将 tokens 转换为对应的 ID
        return [self.vocab.get(token, self.vocab.get(self.unk_token)) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        # 将 IDs 转换为对应的 tokens
        return [self.id_to_token.get(i, self.unk_token) for i in ids]
    

    # def decode(token_ids, tokenizer, skip_special_tokens=True):
    #     tokens = tokenizer.convert_ids_to_tokens(token_ids)
    #     if skip_special_tokens:
    #         # 跳过特殊 token，比如 [PAD], [UNK], [CLS], [SEP]
    #         tokens = [t for t in tokens if t not in {"[PAD]", "[UNK]", "[CLS]", "[SEP]"}]
    #     # 将 tokens 拼接为字符串
    #     return "".join(tokens)


if __name__ == "__main__":
    vocab_file = "./datasets/tokenizer/vocab.json"
    tokenizer = SourceFuzzyTokenizer(vocab_file)

    sequence = "+43.006[274.11782]M+15.995M[869.44292][1420.62591]TTVPC+57.021P[599.30675]"

    tokens = tokenizer.tokenize(sequence)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    print("分词结果:", tokens)
    print("对应的 Token ID:", token_ids)

    reconstructed_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    print("还原 Token:", reconstructed_tokens)
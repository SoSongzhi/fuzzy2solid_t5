import json

# 加载 JSON 数据
def load_mass_combinations(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# 查询指定质量的组合
def query_combinations(mass_dict, target_mass):
    target_mass = str(target_mass)  # 确保键是字符串类型
    return mass_dict.get(target_mass, [])

# 示例使用
if __name__ == "__main__":
    file_path = "../mass_comb_dict.json"  # 替换为实际文件路径
    mass_dict = load_mass_combinations(file_path)

    # 用户输入
    target_mass = input("输入目标质量值（例如 '171.06439'）: ")
    combinations = query_combinations(mass_dict, target_mass)

    if combinations:
        print(f"质量 {target_mass} 对应的组合有: {combinations}")
    else:
        print(f"质量 {target_mass} 没有对应的组合。")
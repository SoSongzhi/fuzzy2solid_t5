import os

# 输出文件名
fuzzy_output_file = "9specie_fuzzy_seqs.txt"
gt_output_file = "9specie_gt_seqs.txt"

# 初始化内容列表
fuzzy_contents = []
gt_contents = []

# 遍历当前文件夹内的所有子文件夹
for root, dirs, files in os.walk("."):
    for file in files:
        if file == "fuzzy_seqs.txt":
            file_path = os.path.join(root, file)
            with open(file_path, "r") as f:
                fuzzy_contents.append(f.read())
        elif file == "gt_seqs.txt":
            file_path = os.path.join(root, file)
            with open(file_path, "r") as f:
                gt_contents.append(f.read())

# 将所有的 fuzzy_seq.txt 内容写入输出文件
with open(fuzzy_output_file, "w") as f:
    f.write("\n".join(fuzzy_contents))

# 将所有的 gt_seq.txt 内容写入输出文件
with open(gt_output_file, "w") as f:
    f.write("\n".join(gt_contents))

print(f"合并完成：{fuzzy_output_file}, {gt_output_file}")
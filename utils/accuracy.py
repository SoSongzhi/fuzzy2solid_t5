def calculate_accuracy(comb, mass, gt_file):
    """
    计算生成序列的逐行准确率（基于生成序列的行数）。

    Args:
        generated_file (str): 生成序列的文件路径，每行一个序列。
        ground_truth_file (str): Ground Truth 序列文件路径，每行一个序列。

    Returns:
        float: 准确率 (0.0 到 1.0)。
    """
    # 读取生成的序列和 Ground Truth 序列
    with open(mass, 'r') as f:
        mass_seqs = [line.rstrip('\n') for line in f]  # 删除每行末尾的换行符，保留空行

    with open(comb, 'r') as f:
        comb_seqs = [line.rstrip('\n') for line in f]  # 删除每行末尾的换行符，保留空行

    with open(gt_file, 'r') as f:
        gseqs = [line.rstrip('\n') for line in f]  # 删除每行末尾的换行符，保留空行

    # 逐行比较，基于 generated_seqs 的行数
    matched_count = 0
    total_count = len(comb_seqs)


    for i, gen_seq in enumerate(comb_seqs):
        # 处理空行的情况：如果生成的序列和 Ground Truth 都是空行，则算作匹配
        if  comb_seqs[i]==gseqs[i]:
            matched_count += 1
        else:
            print("comb:", comb_seqs[i])
            print("mass:", mass_seqs[i])
            print("gt  :", gseqs[i])

    # 计算准确率
    accuracy = matched_count / total_count if total_count > 0 else 0.0

    # 输出结果
    print(f"Generated sequences: {total_count}")
    print(f"comb: {len(comb)}")
    print(f"Matched sequences: {matched_count}")
    print(f"Accuracy: {accuracy:.4f}")

    return accuracy


if __name__ == "__main__":
    # 文件路径
    comb= "../generated_combination.txt"
    mass = "../generated_sequences_based_on_massdic.txt"
    
   
    g = "../validation_gt_seqs.txt"

    # 计算准确率
    calculate_accuracy(comb,mass, g)
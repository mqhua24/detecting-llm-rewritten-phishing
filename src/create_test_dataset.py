import os
import pandas as pd

# 对于测试模型精确度的 测试集 10000 (9:1)

# ---------------------------
# 路径
# ---------------------------
INPUT_PATH = "../fusion_dataset_output/"
OUTPUT_PATH = "../test_dataset_output/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

phishing_file = os.path.join(INPUT_PATH, "phishing_fused.csv")
legit_file = os.path.join(INPUT_PATH, "legit_fused.csv")

# ---------------------------
# 参数
# ---------------------------
TOTAL_SIZE = 10000
RATIO = 0.1  # 钓鱼邮件比例（1:9）
PHISHING_SIZE = int(TOTAL_SIZE * RATIO)
LEGIT_SIZE = TOTAL_SIZE - PHISHING_SIZE

# ---------------------------
# 读取数据
# ---------------------------
phishing_df = pd.read_csv(phishing_file)
legit_df = pd.read_csv(legit_file)

print(f"Phishing emails available: {len(phishing_df)}")
print(f"Legitimate emails available: {len(legit_df)}")

# 如果数据不足，就取能取到的最大数量
phishing_sample_size = min(PHISHING_SIZE, len(phishing_df))
legit_sample_size = min(LEGIT_SIZE, len(legit_df))

# ---------------------------
# 随机抽样
# ---------------------------
phishing_sample = phishing_df.sample(n=phishing_sample_size, random_state=42)
legit_sample = legit_df.sample(n=legit_sample_size, random_state=42)

# ---------------------------
# 合并并打乱
# ---------------------------
test_df = pd.concat([phishing_sample, legit_sample], ignore_index=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# ---------------------------
# 保存
# ---------------------------
output_file = os.path.join(OUTPUT_PATH, "test_dataset.csv")
test_df.to_csv(output_file, index=False)

print(f"Test dataset saved to {output_file}")
print(f"Final dataset size: {len(test_df)} (Phishing: {phishing_sample_size}, Legit: {legit_sample_size})")

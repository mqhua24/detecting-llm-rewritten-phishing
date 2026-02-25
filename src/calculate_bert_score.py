import os

import pandas as pd
import torch
from bert_score import BERTScorer
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TRANSFORMERS_CACHE'] = r"D:/huggingface_cache"

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:1080'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:1080'
# 设置代理地址
proxies = {
    "http": "http://127.0.0.1:1080",
    "https": "http://127.0.0.1:1080"
}
device = "cuda" if torch.cuda.is_available() else "cpu"
# ------------- 1 读取并合并两个文件 Read and merge two files ----------------
file_a = "../fusion_dataset_output/phishing_fused_1000_with_id.csv"   # 原始文本
# file_b = "../together_llama_dataset_output/together_llama_fused_phishing_output_l_bk.csv"   # 改写文本
file_b = "../local_llama_dataset_output/llama3/bertscore1000id/local_llama_fused_phishing_output_m_cleaned.csv"   # 改写文本
col_name = "text"  # 要比较的列名 The column names to be compared

a = pd.read_csv(file_a)
b = pd.read_csv(file_b)

# 按 id 合并，两文件都要有 id Merge by id. Both files must have an id
merged = a.merge(b, on="id", suffixes=("_a", "_b"))

print(f"成功匹配 {len(merged)} 对文本。")

# 上一版 行完全对应可用
# # merged = a.merge(b, on="id", suffixes=("_a", "_b"))
# merged = pd.concat([a.reset_index(drop=True), b.reset_index(drop=True)], axis=1)
# merged.columns = [f"{c}_a" if i < len(a.columns) else f"{c}_b" for i, c in enumerate(list(a.columns) + list(b.columns))]

# print(f"成功匹配 {len(merged)} 对文本。")

# ------------- 2 批量计算BERTScore Batch calculation of BERTScore ----------------
scorer = BERTScorer(
    lang="en",
    rescale_with_baseline=False,
    device=device,
    model_type="roberta-large"
)

P, R, F1 = scorer.score(
    merged[f"{col_name}_b"].tolist(),
    merged[f"{col_name}_a"].tolist()
)

merged["BERTScore_P"] = P
merged["BERTScore_R"] = R
merged["BERTScore_F1"] = F1

mean_val = merged["BERTScore_F1"].mean()

output_dir = "../bertscore_output/local_llama"
# 保存结果 Save the result
merged.to_csv(f"{output_dir}/bertscore_pairwise_results_m.csv", index=False)
print("Saved results: bertscore_pairwise_results.csv")
print("\n Mean BERTScore_F1 =", merged["BERTScore_F1"].mean())

# ------------- 3 可视化部分 Visualization Section ----------------

plt.style.use("seaborn-v0_8-whitegrid")

# ① 分布图（直方图 + KDE） Distribution map (Histogram + KDE)
plt.figure(figsize=(8,5))
sns.histplot(merged["BERTScore_F1"], bins=20, kde=True)
plt.title("Distribution of BERTScore F1", fontsize=14)
plt.xlabel("BERTScore F1")
plt.ylabel("Count")

text_str = f"Mean = {mean_val:.4f}"

plt.text(
    0.98, 0.95,                # 图像坐标(右上角) Image coordinates (upper right corner)
    text_str,
    transform=plt.gca().transAxes,
    ha='right', va='top',
    fontsize=12,
    color='red',
    bbox=dict(
        boxstyle='round',
        edgecolor='black',        # 红色边框
        facecolor='none',       # 不填充背景
        linewidth=1.5
    )
)

plt.tight_layout()
plt.savefig(f"{output_dir}/bertscore_distribution_m.png", dpi=300)
plt.show()

# ② 箱线图（展示离群值与分布形态） Box plot (showing outliers and distribution patterns)
plt.figure(figsize=(6,4))
sns.boxplot(y=merged["BERTScore_F1"])
plt.title("Boxplot of BERTScore F1", fontsize=14)
plt.ylabel("BERTScore F1")
plt.tight_layout()
plt.savefig(f"{output_dir}/bertscore_boxplot_m.png", dpi=300)
plt.show()

# ③ 散点图（原文 vs 改写的F1对应情况）Scatter plot (Original vs rewritten F1 corresponding situation)
plt.figure(figsize=(8,5))
sns.scatterplot(x=range(len(merged)), y=merged["BERTScore_F1"], alpha=0.7)
plt.title("BERTScore F1 per Pair", fontsize=14)
plt.xlabel("Text Pair Index")
plt.ylabel("BERTScore F1")
plt.tight_layout()
plt.savefig(f"{output_dir}/bertscore_scatter_m.png", dpi=300)
plt.show()


# ④ （可选）如果有分类列，比如 level 或 rewrite_type (Optional) If there are classification columns, such as level or rewrite_type
if "level" in merged.columns:
    plt.figure(figsize=(8,5))
    sns.barplot(data=merged, x="level", y="BERTScore_F1", ci="sd")
    plt.title("Average BERTScore F1 by Rewrite Level", fontsize=14)
    plt.xlabel("Rewrite Level")
    plt.ylabel("Mean BERTScore F1")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/bertscore_by_level_m.png", dpi=300)
    plt.show()

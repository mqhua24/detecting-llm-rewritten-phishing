import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 读取两个文件
# df_label = pd.read_csv("../../../qwen3_dataset_output/2000legitandphish/merged_shuffled_qwen3_fused_l_clean_with_id.csv")
# df_spam = pd.read_csv("merged_shuffled_qwen3_fused_l_clean_2000_spam_results_data_clean.csv")

# df_label = pd.read_csv("../../../qwen3_dataset_output/2000legitandphish/merged_shuffled_qwen3_fused_clean_with_id.csv")
# df_spam = pd.read_csv("merged_shuffled_qwen3_fused_clean_2000_spam_results_data_clean.csv")

# df_label = pd.read_csv("../../../qwen3_dataset_output/2000legitandphish/merged_shuffled_fused_original_2000_with_id.csv")
# df_spam = pd.read_csv("merged_shuffled_fused_original_2000_spam_results_data_clean.csv")

# df_label = pd.read_csv("../../../qwen3_dataset_output/2000legitandphish/merged_shuffled_qwen3_fused_m_clean_with_id.csv")
# df_spam = pd.read_csv("merged_shuffled_qwen3_fused_m_clean_2000_spam_results_data_clean.csv")

# df_label = pd.read_csv("../../../gpt_dataset_output/2000legitandphish/merged_shuffled_gpt_fused_m_with_id.csv")
# df_spam = pd.read_csv("merged_shuffled_gpt_fused_m_2000_spam_results_data_clean.csv")

# df_label = pd.read_csv("../../../gpt_dataset_output/2000legitandphish/merged_shuffled_gpt_fused_l_with_id.csv")
# df_spam = pd.read_csv("merged_shuffled_gpt_fused_l_2000_spam_results_data_clean.csv")

# df_label = pd.read_csv("../../../gpt_dataset_output/2000legitandphish/merged_shuffled_gpt_fused_with_id.csv")
# df_spam = pd.read_csv("merged_shuffled_gpt_fused_2000_spam_results_data_clean.csv")

# df_label = pd.read_csv("../../../together_llama_dataset_output/2000legitandphish/merged_shuffled_together_llama_fused_phishing_output_m_with_id.csv")
# df_spam = pd.read_csv("merged_shuffled_together_llama_fused_m_2000_spam_results_data_clean.csv")

df_label = pd.read_csv("../../../local_llama_dataset_output/llama3/2000legitandphish/merged_shuffled_local_llama_fused_phishing_output_cleaned_with_id.csv")
df_spam = pd.read_csv("merged_shuffled_local_llama_fused_2000_spam_results_data_clean.csv")

# ✅ 把Email列重命名为id
df_spam.rename(columns={"Email": "id"}, inplace=True)

# 按id匹配（只保留检测成功的样本）
df_merged = pd.merge(df_label, df_spam, on="id", how="inner")

# 将 True/False 转换为 1/0
df_merged["spam_pred"] = df_merged["Spam"].map({True: 1, False: 0})

# # ✅ 保存合并后的结果
# df_merged.to_csv("pred/merged_shuffled_qwen3_fused_l_clean_2000_spam_results_data_clean_with_predictions.csv", index=False)
#
# print("✅ 已保存合并结果到 emails_with_predictions.csv")
# print(df_merged.head())
# print(f"样本总数: {len(df_merged)}")

# ===== 计算准确率、召回率等 =====
y_true = df_merged["label"]
y_pred = df_merged["spam_pred"]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n📊 模型评估结果：")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\n详细报告：")
print(classification_report(y_true, y_pred, target_names=["Normal", "Spam"]))
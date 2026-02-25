import pandas as pd

# 两个数据集
# df1 = pd.read_csv("../merged_shuffled_qwen3_fused_clean_with_id_detect_result_process.csv")
# df2 = pd.read_csv("../../../../qwen3_dataset_output/2000legitandphish/merged_shuffled_qwen3_fused_clean_with_id.csv")

# df1 = pd.read_csv("../merged_shuffled_gpt_fused_with_id_detect_result_process.csv")
# df2 = pd.read_csv("../../../../gpt_dataset_output/2000legitandphish/merged_shuffled_gpt_fused_with_id.csv")

# df1 = pd.read_csv("../merged_shuffled_local_llama_fused_phishing_output_gate_result_process.csv")
# df2 = pd.read_csv("../../../../local_llama_dataset_output/llama3/2000legitandphish/merged_shuffled_local_llama_fused_phishing_output_cleaned_with_id.csv")

#
# # 按 id 合并，两份数据中都必须有 id
# merged = pd.merge(df1[['id', 'label']], df2[['id', 'label']], on='id', suffixes=('_1', '_2'))
#
# # 比较 label_1 和 label_2
# correct = (merged['label_1'] == merged['label_2']).sum()
# accuracy = correct / len(merged)
#
# print("对齐后的总数量：", len(merged))
# print("一致数量：", correct)
# print("准确率：", accuracy)


# tp = ((merged['risk_label'] == 1) & (merged['label'] == 1)).sum()
# fn = ((merged['risk_label'] == 1) & (merged['label'] == 0)).sum()
#
# recall = tp / (tp + fn) if (tp + fn) > 0 else 0
# print("召回率 Recall:", recall)


# import pandas as pd
#
# # 按 id 合并
# merged = pd.merge(
#     df1[['id', 'risk_label']],
#     df2[['id', 'label']],
#     on='id',
#     how='inner'
# )
#
# # # 二者比较（df1 为真值）
# # y_true = merged['risk_label']
# # y_pred = merged['label']
#
# # 真值来自 df2
# y_true = merged['label']          # 正确标签
# y_pred = merged['risk_label']     # 模型预测
#
# # 准确率
# accuracy = (y_true == y_pred).mean()
#
# # 召回率 (Recall = TP / (TP + FN))
# # 以 risk_label == 1 作为正类
# tp = ((y_true == 1) & (y_pred == 1)).sum()
# fn = ((y_true == 1) & (y_pred == 0)).sum()
# recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#
# print("对齐后的总数量：", len(merged))
# print("准确率 Accuracy:", accuracy)
# print("召回率 Recall:", recall)




df1 = pd.read_csv("../merged_shuffled_local_llama_fused_phishing_output_l_new2_gate_result_process.csv")
df2 = pd.read_csv("../../../../local_llama_dataset_output/llama3/2000legitandphish/merged_shuffeld_local_llama_fused_phishing_output_l_new2_cleaned_with_id.csv")
import pandas as pd

merged = pd.merge(
    df1[['id', 'risk_label']],   # 预测
    df2[['id', 'label']],        # 真值
    on='id',
    how='inner'
)

y_true = merged['label'].astype(int)
y_pred = merged['risk_label'].astype(int)

# 基本指标
accuracy = (y_true == y_pred).mean()

tp = ((y_true == 1) & (y_pred == 1)).sum()
tn = ((y_true == 0) & (y_pred == 0)).sum()
fp = ((y_true == 0) & (y_pred == 1)).sum()
fn = ((y_true == 1) & (y_pred == 0)).sum()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("对齐后的总数量：", len(merged))
print("准确率 Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
print("混淆矩阵: TP, FP, TN, FN =", tp, fp, tn, fn)
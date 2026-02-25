import pandas as pd

# 两个数据集
# df1 = pd.read_csv("../merged_shuffled_local_llama_fused_phishing_output_detect_result_process.csv")
# df2 = pd.read_csv("../../../../local_llama_dataset_output/llama3/2000legitandphish/merged_shuffled_local_llama_fused_phishing_output_cleaned_with_id.csv")

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


df1 = pd.read_csv("../merged_shuffled_fused_original_2000_detect_result_process.csv")
df2 = pd.read_csv("../../../../qwen3_dataset_output/2000legitandphish/merged_shuffled_fused_original_2000_with_id.csv")


# 按 id 合并
merged = pd.merge(
    df1[['id', 'label']],
    df2[['id', 'label']],
    on='id',
    suffixes=('_1', '_2')   # df1 是 label_1（预测）, df2 是 label_2（真值）
)

# 定义真值与预测
y_pred = merged['label_1'].astype(int)
y_true = merged['label_2'].astype(int)

# 准确率
accuracy = (y_pred == y_true).mean()

# 混淆矩阵
tp = ((y_pred == 1) & (y_true == 1)).sum()
tn = ((y_pred == 0) & (y_true == 0)).sum()
fp = ((y_pred == 1) & (y_true == 0)).sum()
fn = ((y_pred == 0) & (y_true == 1)).sum()

# Precision, Recall, F1
precision = tp / (tp + fp) if tp + fp > 0 else 0
recall = tp / (tp + fn) if tp + fn > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

# 输出结果
print("对齐后的总数量：", len(merged))
print("一致数量 correct：", (y_pred == y_true).sum())
print("准确率 Accuracy:", accuracy)
print("召回率 Recall:", recall)
print("精确率 Precision:", precision)
print("F1 分数:", f1)
print("混淆矩阵: TP, FP, TN, FN =", tp, fp, tn, fn)

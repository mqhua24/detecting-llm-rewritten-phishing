import os
import pandas as pd

# ---------------------------
# 路径
# ---------------------------
INPUT_PATH = "../filter_dataset_output/"  # 之前过滤后的 CSV
OUTPUT_PATH = "../fusion_dataset_output/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 输入文件列表
phishing_file = os.path.join(INPUT_PATH, "phishing_filtered.csv")
legit_file = os.path.join(INPUT_PATH, "legit_filtered.csv")

# ---------------------------
# 融合函数
# ---------------------------
def fuse_columns(df):
    # 检查列是否存在
    for col in ['sender', 'receiver', 'date', 'subject', 'body']:
        if col not in df.columns:
            df[col] = ''
    # 合并为 text 并加前缀
    def merge_row(row):
        parts = [
            f"sender: {str(row.get('sender', '')).strip()}",
            f"receiver: {str(row.get('receiver', '')).strip()}",
            f"date: {str(row.get('date', '')).strip()}",
            f"subject: {str(row.get('subject', '')).strip()}",
            str(row.get('body', '')).strip()
        ]
        return "\n".join(parts)
    df['text'] = df.apply(merge_row, axis=1)
    # 保留 text、label、urls
    if 'urls' not in df.columns:
        df['urls'] = ''
    return df[['text', 'label', 'urls']]

# ---------------------------
# 处理钓鱼邮件
# ---------------------------
phishing_df = pd.read_csv(phishing_file)
phishing_fused = fuse_columns(phishing_df)
phishing_fused.to_csv(os.path.join(OUTPUT_PATH, "phishing_fused.csv"), index=False)

# ---------------------------
# 处理正常邮件
# ---------------------------
legit_df = pd.read_csv(legit_file)
legit_fused = fuse_columns(legit_df)
legit_fused.to_csv(os.path.join(OUTPUT_PATH, "legit_fused.csv"), index=False)

# 融合列邮件已保存到 fusion_dataset_output/ 目录
print("The fusion column mail has been saved to the fusion_dataset_output/ directory")

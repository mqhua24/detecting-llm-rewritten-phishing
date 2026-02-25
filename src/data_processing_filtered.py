import os
import pandas as pd
import re

# ---------------------------
# 路径
# ---------------------------
DATA_PATH = "../dataset/"
OUTPUT_PATH = "../filter_dataset_output/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

nazario_file = os.path.join(DATA_PATH, "Nazario.csv")
nigerian_file = os.path.join(DATA_PATH, "Nigerian_Fraud.csv")
enron_file = os.path.join(DATA_PATH, "Enron.csv")
# trec_files = [os.path.join(DATA_PATH, f"TREC_{i:02}.csv") for i in [5, 6, 7]]
spamassassin_file = os.path.join(DATA_PATH, "SpamAssasin.csv")  # 改文件名

# ---------------------------
# 英文文本判断函数
# ---------------------------
def is_english(text):
    if pd.isna(text) or len(text.strip()) == 0:
        return False
    # 简单英文单词占比判断
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
    return len(words) / max(len(text.split()), 1) > 0.5

# ---------------------------
# 标签过滤函数
# ---------------------------
def filter_by_label(df, label_value):
    if 'label' not in df.columns:
        return pd.DataFrame()  # 防止列不存在报错
    return df[df['label'] == label_value].copy()

# ---------------------------
# 数据处理函数
# ---------------------------
def process_file(file_path, label_value=None, english_filter=False):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path, encoding="utf-8", on_bad_lines='skip')
        print(f"{file_path} read {len(df)} rows")
        if label_value is not None:
            df = filter_by_label(df, label_value)
        if english_filter and 'body' in df.columns:
            df = df[df['body'].apply(is_english)]
        print(f"{file_path} after filtering: {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()

# ---------------------------
# 钓鱼邮件
# ---------------------------
phishing_dfs = []
for f in [nazario_file, nigerian_file]:
    phishing_dfs.append(process_file(f, label_value=1, english_filter=True))
phishing_df = pd.concat(phishing_dfs, ignore_index=True) if phishing_dfs else pd.DataFrame()
print(f"Total phishing emails: {len(phishing_df)}")

# ---------------------------
# 正常邮件
# ---------------------------
legit_dfs = []
for f in [enron_file] + [spamassassin_file]:
    legit_dfs.append(process_file(f, label_value=0, english_filter=True))
legit_df = pd.concat(legit_dfs, ignore_index=True) if legit_dfs else pd.DataFrame()
print(f"Total legitimate emails: {len(legit_df)}")

# ---------------------------
# 保存到 output 目录
# ---------------------------
phishing_df.to_csv(os.path.join(OUTPUT_PATH, "phishing_filtered.csv"), index=False)
legit_df.to_csv(os.path.join(OUTPUT_PATH, "legit_filtered.csv"), index=False)
print("The filtered version of the email has been saved to the filter_dataset_output/ directory")  #过滤版邮件已保存到 filter_dataset_output/ 目录

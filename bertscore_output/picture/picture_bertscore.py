import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

sns.set(style="whitegrid", font_scale=1.2)

# ------------------------------
# STEP 1：读取所有 CSV 文件
# STEP 1: Read all CSV files
# ------------------------------
files = glob.glob("allbertscore/*.csv")  # 替换成你的路径 Replace it with your path
df_all = []

for f in files:
    dataset_name = os.path.basename(f).replace(".csv", "")
    df = pd.read_csv(f)

    # 提取 BERTScore_F1
    df = df[['BERTScore_F1']].copy()

    # 添加 dataset 列
    df['dataset'] = dataset_name
    df_all.append(df)

df_all = pd.concat(df_all, ignore_index=True)

# ------------------------------
# STEP 2：计算平均值和标准差
# STEP 2: Calculate the mean and standard deviation
# ------------------------------
summary = df_all.groupby('dataset')['BERTScore_F1'].agg(['mean', 'std']).reset_index()
summary = summary.sort_values('mean', ascending=False)

# ------------------------------
# STEP 3：颜色映射
# STEP 3: Color Mapping
# ------------------------------
palette = sns.color_palette("tab10", n_colors=len(summary))
color_dict = dict(zip(summary['dataset'], palette))

# ------------------------------
# STEP 4：Boxplot + 平均值标记
# STEP 4: Boxplot + Average marking
# ------------------------------
plt.figure(figsize=(14, 6))
ax = sns.boxplot(x='dataset', y='BERTScore_F1', data=df_all, palette=color_dict, showfliers=True)

# 标注平均值
for i, row in enumerate(summary.itertuples()):
    plt.scatter(i, row.mean, color='red', s=50, zorder=10, marker='D', label="Mean" if i == 0 else "")

plt.xticks(rotation=45, ha='right')
plt.ylabel("BERTScore F1")
plt.title("BERTScore Distribution Across All Datasets (Boxplot + Mean)")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------
# STEP 5：Violin plot + 平均值
# STEP 5: Violin plot + average value
# ------------------------------
plt.figure(figsize=(14, 6))
ax = sns.violinplot(x='dataset', y='BERTScore_F1', data=df_all, inner=None, palette=color_dict)

for i, row in enumerate(summary.itertuples()):
    plt.scatter(i, row.mean, color='red', s=50, zorder=10, marker='D', label="Mean" if i == 0 else "")

plt.xticks(rotation=45, ha='right')
plt.ylabel("BERTScore F1")
plt.title("BERTScore Distribution Across All Datasets (Violin + Mean)")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------
# STEP 6：KDE 密度曲线
# STEP 6: KDE density curve
# ------------------------------
plt.figure(figsize=(12, 6))
for name, group in df_all.groupby('dataset'):
    sns.kdeplot(group['BERTScore_F1'], label=f"{name} (mean={group['BERTScore_F1'].mean():.3f})",
                linewidth=2, fill=True, alpha=0.3, color=color_dict[name])

plt.xlabel("BERTScore F1")
plt.ylabel("Density")
plt.title("BERTScore Density Across All Datasets (KDE)")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------
# STEP 7：平均 BERTScore 柱状图 + 标准差
# STEP 7: Average BERTScore Bar Chart + Standard Deviation
# ------------------------------
plt.figure(figsize=(14, 6))
plt.bar(summary['dataset'], summary['mean'], yerr=summary['std'], color=palette, capsize=6)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Average BERTScore F1")
plt.title("Average BERTScore per Dataset (with Std Dev)")
plt.tight_layout()
plt.show()

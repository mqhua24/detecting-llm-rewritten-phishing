import re
from pathlib import Path
import pandas as pd
import numpy as np

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import matplotlib.pyplot as plt

# =============== 1) 配置：文件与标签 Configuration: Files and tags ===============
# 说明：accuracy = ElSlay；ealv = ealvaradob
FILES = {
    # Llama-3.3-70B
    "factors_llama3.3_70b_accuracy.xlsx":     {"LLM": "Llama-3.3-70B", "Detector": "ElSlay"},
    "factors_llama3.3_70b_ealv_accuracy.xlsx":{"LLM": "Llama-3.3-70B", "Detector": "ealvaradob"},
    # Qwen3-4B
    "factors_qwen3_4b_q8_accuracy.xlsx":      {"LLM": "Qwen-3-4B",     "Detector": "ElSlay"},
    "factors_qwen3_4b_q8_ealv_accuracy.xlsx": {"LLM": "Qwen-3-4B",     "Detector": "ealvaradob"},
    # Qwen3-30B
    "factors_qwen3_30b_ali_accuracy.xlsx":    {"LLM": "Qwen-3-30B",    "Detector": "ElSlay"},
    "factors_qwen3_30b_ali_ealv_accuracy.xlsx":{"LLM": "Qwen-3-30B",   "Detector": "ealvaradob"},
}

BASE_DIR = Path(".")  # 如文件不在当前目录，可改成目录路径 If the file is not in the current directory, it can be changed to a directory path


# =============== 2) 工具函数：稳健读取与清洗 Utility function: Robust reading and cleaning ===============
def clean_columns(cols):
    """
    标准化列名：去空格、统一小写、去掉重复空格/括号内容等轻度清洗。
    Standardized column names: removing Spaces, unifying lowercase letters, and eliminating duplicate Spaces/parentheses, etc. for light cleaning.
    """
    new = []
    for c in cols:
        c = str(c).strip()
        if c.lower().startswith("unnamed"):
            new.append(None)  # 标记等会儿丢掉
            continue
        # 去除多余空白 Remove unnecessary blank Spaces
        c = re.sub(r"\s+", " ", c)
        new.append(c)
    return new

def find_header_row(df):
    """
    如果第一行并不是我们想要的表头（比如真正表头出现在第k行），
    就找包含关键字段的那一行作为表头。
    If the first row is not the header we want (for example, the actual header appears in the KTH row),
    Just find the row containing the key fields as the header of the table.
    """
    key_candidates = {"combo_id", "cta", "amount", "sensitive", "accuracy"}
    for i in range(min(10, len(df))):  # 仅在前10行里找表头 Look for the header only in the first 10 lines
        row_vals = set(str(x).strip().lower() for x in df.iloc[i].tolist())
        overlap = key_candidates & row_vals
        if len(overlap) >= 3:
            return i
    return 0  # 默认第一行就是表头 By default, the first line is the header of the table

def load_one_excel(path: Path, llm: str, detector: str) -> pd.DataFrame:
    # 第一次按默认表头读 Read the default header for the first time
    df0 = pd.read_excel(path, engine="openpyxl", header=0)
    # 有些文件可能是“多行表头”，尝试定位真正表头行 Some files might be "multi-line headers". Try to locate the actual header lines
    hdr = find_header_row(df0)
    if hdr != 0:
        df = pd.read_excel(path, engine="openpyxl", header=hdr)
    else:
        df = df0.copy()

    # 列名清洗 List cleaning
    cols = clean_columns(df.columns)
    df.columns = cols
    # 丢掉 None 列（原来的 Unnamed）Discard the None column (the original Unnamed)
    df = df.loc[:, [c for c in df.columns if c]]

    # 统一小写列名（留一份映射，等会儿重命名）Unify the lowercase column names (keep a mapping and rename it later)
    lower_map = {c: c.lower() for c in df.columns}
    df.rename(columns=lower_map, inplace=True)

    # 只保留可用列（如果存在）Only retain the available columns (if they exist)
    keep_candidates = ["combo_id", "cta", "amount", "sensitive", "accuracy"]
    keep = [c for c in keep_candidates if c in df.columns]
    # 如果没有 accuracy 列，但出现类似 "overall average accuracy rate" 这种垃圾列，直接忽略
    # If there is no "accuracy" column but a junk column like "overall average accuracy rate" appears, simply ignore it
    if "accuracy" not in keep and "accuracy" in df.columns:
        keep.append("accuracy")
    df = df[keep]

    # 去掉明显的汇总/空行：accuracy 转数值，非数值行丢弃
    # Remove obvious summary/blank lines: Convert accuracy to numerical values and discard non-numerical lines
    if "accuracy" in df.columns:
        df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")

    # 去掉完全空的行 & 没有 accuracy 的行 Remove completely empty lines and lines without accuracy
    df = df.dropna(how="all")
    if "accuracy" in df.columns:
        df = df.dropna(subset=["accuracy"])

    # 标注 LLM / Detector Mark LLM/Detector
    df["LLM"] = llm
    df["Detector"] = detector

    # 将分类因子统一为字符串（避免 statsmodels 把 0/1 当数值）
    # Unify the classification factors as strings (to avoid statsmodels treating 0/1 as numerical values)
    for cat_col in ["cta", "amount", "sensitive", "LLM", "Detector"]:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype(str).str.strip()

    return df


# =============== 3) 合并所有文件 Merge all files ===============
all_dfs = []
for fname, meta in FILES.items():
    fpath = BASE_DIR / fname
    if not fpath.exists():
        print(f"[WARN] file does not exist: {fpath}")
        continue
    one = load_one_excel(fpath, meta["LLM"], meta["Detector"])
    all_dfs.append(one)

data = pd.concat(all_dfs, ignore_index=True)
print("== Cleaned columns ==", list(data.columns))
print("shape:", data.shape)
print(data.head(3))


# =============== 4) 统计分析（ANOVA） ===============
# 统一把 accuracy -> Accuracy（仅为公式美观，非必需）
if "accuracy" not in data.columns:
    raise RuntimeError("未找到 accuracy 列，请检查文件内容。")
data = data.rename(columns={"accuracy": "Accuracy"})

# —— 双因素：LLM × Detector two-factor
print("\n===== Two-way ANOVA: Accuracy ~ C(LLM) * C(Detector) =====")
model2 = ols('Accuracy ~ C(LLM) * C(Detector)', data=data).fit()
anova2 = sm.stats.anova_lm(model2, typ=2)
print(anova2)
print(model2.summary())

# —— 三因素（如果有 CTA）Three factors (if there is a CTA)
if "cta" in data.columns:
    print("\n===== Three-way ANOVA: Accuracy ~ C(LLM) * C(Detector) * C(CTA) =====")
    model3 = ols('Accuracy ~ C(LLM) * C(Detector) * C(cta)', data=data).fit()
    anova3 = sm.stats.anova_lm(model3, typ=2)
    print(anova3)


import seaborn as sns

plt.figure(figsize=(6,4))
sns.boxplot(x='cta', y='Accuracy', data=data, palette='Set2')
sns.pointplot(x='cta', y='Accuracy', data=data, color='black', markers='D', linestyles='--', errorbar='sd')
plt.title('Detection accuracy across CTA levels')
plt.xlabel('CTA level (E: Explicit, S: Softened, H: Hidden)')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()


if "amount" in data.columns:
    print("\n===== Three-way ANOVA: Accuracy ~ C(LLM) * C(Detector) * C(amount) =====")
    model4 = ols('Accuracy ~ C(LLM) * C(Detector) * C(amount)', data=data).fit()
    anova4 = sm.stats.anova_lm(model4, typ=2)
    print(anova4)

if "sensitive" in data.columns:
    print("\n===== Three-way ANOVA: Accuracy ~ C(LLM) * C(Detector) * C(sensitive) =====")
    model5 = ols('Accuracy ~ C(LLM) * C(Detector) * C(sensitive)', data=data).fit()
    anova5 = sm.stats.anova_lm(model5, typ=2)
    print(anova5)

# —— 可选：对 LLM 做 Tukey 事后检验
# -- Optional: Perform Tukey post hoc checks on LLMS
print("\n===== Tukey HSD on LLM (by Accuracy) =====")
tukey = pairwise_tukeyhsd(endog=data["Accuracy"], groups=data["LLM"], alpha=0.05)
print(tukey.summary())

# —— 可选：对 cta 做 Tukey 事后检验
# -- Optional: Conduct a Tukey post hoc check on the cta
print("\n===== Tukey HSD on CTA(by Accuracy) =====")
tukey = pairwise_tukeyhsd(endog=data["Accuracy"], groups=data["cta"], alpha=0.05)
print(tukey.summary())


# =============== 5) 可视化（matplotlib） ===============
# 箱线图：不同 LLM × Detector 的 Accuracy 分布 Box plot: Accuracy distribution of different LLMS × Detectors
plt.figure(figsize=(9, 5))
# 不用 seaborn，纯 matplotlib
# 为了简单，按 Detector 分两个子集画
for i, det in enumerate(sorted(data["Detector"].unique())):
    sub = data[data["Detector"] == det]
    groups = [sub[sub["LLM"] == llm]["Accuracy"].values
              for llm in sorted(data["LLM"].unique())]
    ax = plt.subplot(1, len(data["Detector"].unique()), i+1)
    ax.boxplot(groups, labels=sorted(data["LLM"].unique()))
    ax.set_title(f"Detector: {det}")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("LLM")

plt.tight_layout()
plt.show()

# 热力图：LLM × Detector 的均值可视化 Heat map: LLM × Detector mean visualization
pivot = data.pivot_table(index="LLM", columns="Detector", values="Accuracy", aggfunc="mean")
plt.figure(figsize=(5, 4))
plt.imshow(pivot.values, aspect="auto")
plt.colorbar(label="Mean Accuracy")
plt.xticks(range(pivot.shape[1]), pivot.columns, rotation=45, ha="right")
plt.yticks(range(pivot.shape[0]), pivot.index)
plt.title("Mean Accuracy by LLM × Detector")
plt.tight_layout()
plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd

sns.set(style="whitegrid")           # 论文友好风格
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})

# 假设 data 已经是你前面拼好的 DataFrame，列名 "Accuracy","LLM","Detector","cta","amount","sensitive"
# ---------- CTA box + point plot with Tukey annotations ----------
# def plot_cta_with_stats(data, outpath='fig_cta_accuracy.png'):
#     import math
#     # 期望的类别顺序（若数据中没有某些类别也能正常处理）
#     cta_order = ['E', 'S', 'H']
#
#     # 计算每组的统计量（mean, n, std）
#     stats = data.groupby('cta')['Accuracy'].agg(['mean', 'count', 'std']).rename(columns={'count':'n'})
#     # 确保包含所有希望的行以便后续索引稳定
#     stats = stats.reindex(cta_order)
#     stats['n'] = stats['n'].fillna(0).astype(int)
#     stats['std'] = stats['std'].fillna(0.0)
#
#     # 标准误与 95% CI（正态近似；样本小可改为 t 分布）
#     stats['se'] = stats.apply(lambda r: (r['std'] / math.sqrt(r['n'])) if r['n'] > 0 else 0.0, axis=1)
#     stats['ci95'] = 1.96 * stats['se']
#
#     # 绘图准备
#     plt.figure(figsize=(6.2,4.4))
#     ax = plt.gca()
#
#     # 画箱线图（只使用 seaborn 的箱线部分，order 仍为 cta_order，让缺失类别显示为空）
#     sns.boxplot(x='cta', y='Accuracy', data=data, order=cta_order,
#                 showcaps=True, boxprops={'facecolor':'#d9e6f2'},
#                 medianprops={'color':'black'}, showfliers=False, ax=ax)
#
#     # 使用 matplotlib 绘制均值与 95% CI（避免 seaborn.pointplot 的兼容性问题）
#     x_positions = [cta_order.index(c) for c in cta_order]
#     means = [stats.loc[c, 'mean'] if not np.isnan(stats.loc[c, 'mean']) else np.nan for c in cta_order]
#     cis = [stats.loc[c, 'ci95'] for c in cta_order]
#     ns = [int(stats.loc[c, 'n']) for c in cta_order]
#
#     for xi, m, ci, n in zip(x_positions, means, cis, ns):
#         if n > 0 and not np.isnan(m):
#             ax.errorbar(x=xi, y=m, yerr=ci, fmt='D', markersize=6, elinewidth=1.2, capsize=4, zorder=5)
#             ax.text(xi, m + ci + 0.01 * (data['Accuracy'].max() - data['Accuracy'].min()), f"{m:.3f}",
#                     ha='center', va='bottom', fontsize=9)
#
#     # 在 x 轴下方标注样本量
#     ylim = ax.get_ylim()
#     y_min = ylim[0]
#     for xi, n in zip(x_positions, ns):
#         ax.text(xi, y_min - 0.03 * (ylim[1] - ylim[0]), f"n={n}", ha='center', va='top', fontsize=9)
#
#     # Tukey HSD：加保护以免因数据问题抛错
#     sig_pairs = []
#     try:
#         nonzero_groups = [c for c in cta_order if stats.loc[c, 'n'] > 0]
#         if len(nonzero_groups) >= 2:
#             tukey = pairwise_tukeyhsd(endog=data["Accuracy"], groups=data["cta"], alpha=0.05)
#             for row in tukey.summary().data[1:]:
#                 g1, g2, meandiff, p_adj, lower, upper, reject = row
#                 if reject:
#                     sig_pairs.append((g1, g2, float(p_adj)))
#     except Exception as e:
#         print(f"[WARN] Tukey HSD failed: {e}")
#
#     # 绘制显著性标注（若有）
#     if sig_pairs:
#         y_max = data['Accuracy'].max()
#         step = (y_max - data['Accuracy'].min()) * 0.06 if (y_max - data['Accuracy'].min()) > 0 else 0.05
#         start_y = y_max + step
#         for i, (g1, g2, pval) in enumerate(sig_pairs):
#             if g1 not in cta_order or g2 not in cta_order:
#                 continue
#             x1 = cta_order.index(g1)
#             x2 = cta_order.index(g2)
#             y = start_y + i * step
#             ax.plot([x1, x1, x2, x2], [y-0.005, y, y, y-0.005], lw=1.2, color='black')
#             if pval < 0.001:
#                 txt = '***'
#             elif pval < 0.01:
#                 txt = '**'
#             elif pval < 0.05:
#                 txt = '*'
#             else:
#                 txt = f"p={pval:.2f}"
#             ax.text((x1 + x2) / 2, y + step * 0.02, txt, ha='center', va='bottom', fontsize=10)
#         ax.set_ylim(bottom=data['Accuracy'].min() - 0.06 * (ylim[1] - ylim[0]), top=start_y + step * 1.8)
#
#     ax.set_xlabel('CTA level (E: Explicit, S: Softened, H: Hidden)')
#     ax.set_ylabel('Detection Accuracy')
#     ax.set_title('Detection accuracy across CTA levels (all LLMs & detectors)')
#     plt.tight_layout()
#     plt.savefig(outpath, dpi=300)
#     plt.close()

def plot_cta_with_stats(data, outpath='fig_cta_accuracy.png'):
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    # 期望的类别顺序（若数据中没有某些类别也能正常处理）
    # The expected category order (if there are no certain categories in the data, it can still be processed normally)
    cta_order = ['E', 'S', 'H']

    # 计算每组的统计量（mean, n, std）Calculate the statistics (mean, n, std) of each group
    stats = data.groupby('cta')['Accuracy'].agg(['mean', 'count', 'std']).rename(columns={'count':'n'})
    stats = stats.reindex(cta_order)
    stats['n'] = stats['n'].fillna(0).astype(int)
    stats['std'] = stats['std'].fillna(0.0)

    # 标准误与 95% CI
    stats['se'] = stats.apply(lambda r: (r['std'] / math.sqrt(r['n'])) if r['n'] > 0 else 0.0, axis=1)
    stats['ci95'] = 1.96 * stats['se']

    # 绘图准备
    plt.figure(figsize=(6.2,4.4))
    ax = plt.gca()

    # 画箱线图
    sns.boxplot(x='cta', y='Accuracy', data=data, order=cta_order,
                showcaps=True, boxprops={'facecolor':'#d9e6f2'},
                medianprops={'color':'black'}, showfliers=False, ax=ax)

    # 绘制均值与 95% CI
    x_positions = [cta_order.index(c) for c in cta_order]
    means = [stats.loc[c, 'mean'] if not np.isnan(stats.loc[c, 'mean']) else np.nan for c in cta_order]
    cis = [stats.loc[c, 'ci95'] for c in cta_order]
    ns = [int(stats.loc[c, 'n']) for c in cta_order]

    for xi, m, ci, n in zip(x_positions, means, cis, ns):
        if n > 0 and not np.isnan(m):
            # ax.errorbar(x=xi, y=m, yerr=ci, fmt='D', markersize=6, elinewidth=1.2, capsize=4, zorder=5)
            ax.errorbar(x=xi, y=m, yerr=ci if ci > 0 else 0.0,
                        fmt='D', markersize=7, elinewidth=2.0, capsize=5,
                        ecolor='black',  # 误差线颜色
                        markeredgecolor='black', markeredgewidth=1.0,
                        markerfacecolor='white', zorder=10)

            ax.text(xi, m + ci + 0.01 * (data['Accuracy'].max() - data['Accuracy'].min()), f"{m:.3f}",

                    ha='center', va='bottom', fontsize=9)

    # 在 x 轴下方标注样本量 n，避免和横坐标标签重叠
    for xi, n in zip(x_positions, ns):
        ax.annotate(f"n={n}",
                    xy=(xi, 0),  # 基于 x 轴 y=0
                    xytext=(0, -20),  # 向下偏移 20 点
                    textcoords='offset points',
                    ha='center', va='top', fontsize=9)

    # Tukey HSD
    sig_pairs = []
    try:
        nonzero_groups = [c for c in cta_order if stats.loc[c, 'n'] > 0]
        if len(nonzero_groups) >= 2:
            tukey = pairwise_tukeyhsd(endog=data["Accuracy"], groups=data["cta"], alpha=0.05)
            for row in tukey.summary().data[1:]:
                g1, g2, meandiff, p_adj, lower, upper, reject = row
                if reject:
                    sig_pairs.append((g1, g2, float(p_adj)))
    except Exception as e:
        print(f"[WARN] Tukey HSD failed: {e}")

    # 绘制显著性标注（直接显示 p 值） Draw saliency annotations (directly display the p-value)
    if sig_pairs:
        y_max = data['Accuracy'].max()
        step = (y_max - data['Accuracy'].min()) * 0.06 if (y_max - data['Accuracy'].min()) > 0 else 0.05
        start_y = y_max + step
        for i, (g1, g2, pval) in enumerate(sig_pairs):
            if g1 not in cta_order or g2 not in cta_order:
                continue
            x1 = cta_order.index(g1)
            x2 = cta_order.index(g2)
            y = start_y + i * step
            ax.plot([x1, x1, x2, x2], [y-0.005, y, y, y-0.005], lw=1.2, color='black')
            # 根据大小选择显示格式
            if pval < 1e-10:
                p_text = "p<1e-10"  # 对极小 p 值直接标注小于某阈值
            else:
                p_text = f"p={pval:.2e}"  # 否则显示科学计数法
            ax.text((x1 + x2) / 2, y + step * 0.02, p_text, ha='center', va='bottom', fontsize=10)

            #ax.text((x1 + x2) / 2, y + step * 0.02, f"p={pval:.2e}", ha='center', va='bottom', fontsize=10)

        ax.set_ylim(bottom=data['Accuracy'].min() - 0.06 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                    top=start_y + len(sig_pairs) * step + step)

    ax.set_xlabel('CTA level (E: Explicit, S: Softened, H: Hidden)')
    ax.set_ylabel('Detection Accuracy')
    ax.set_title('Detection accuracy across CTA levels (all LLMs & detectors)')
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_by_llm_detector_boxes(data, outpath='fig_llm_detector_boxes.png'):
    """
    Plot Accuracy by LLM for each Detector.
    - Boxplots (semi-transparent) for distribution
    - Mean with 95% CI as diamond markers
    - Tukey HSD p-value annotations
    """
    import math
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    sns.set(style="whitegrid")
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11})

    detectors = sorted(data['Detector'].unique())
    llms = sorted(data['LLM'].unique())
    n_det = len(detectors)
    fig_w = max(6.4 * n_det, 6.4)
    fig, axes = plt.subplots(1, n_det, figsize=(fig_w, 4), sharey=True)
    if n_det == 1:
        axes = [axes]

    palette = sns.color_palette("pastel", n_colors=len(llms))
    color_map = {llm: palette[i] for i, llm in enumerate(llms)}

    for ax, det in zip(axes, detectors):
        sub = data[data['Detector'] == det].copy()

        # Boxplot with semi-transparent boxes
        sns.boxplot(
            x='LLM', y='Accuracy', data=sub, order=llms, ax=ax,
            showcaps=True, medianprops={'color':'black'}, showfliers=False, width=0.6,
            boxprops={'facecolor':'#d9e6f2', 'alpha':0.5},
            whiskerprops={'color':'black'},
            capprops={'color':'black'}
        )

        # Optional: fill boxes with custom colors
        try:
            for patch, llm in zip(ax.artists, llms):
                col = color_map[llm]
                patch.set_facecolor(col)
                patch.set_alpha(0.5)
                patch.set_edgecolor('black')
        except Exception:
            pass

        # Compute mean, std, CI
        stats = sub.groupby('LLM')['Accuracy'].agg(['mean', 'count', 'std']).reindex(llms)
        stats['count'] = stats['count'].fillna(0).astype(int)
        stats['std'] = stats['std'].fillna(0.0)
        stats['se'] = stats.apply(lambda r: (r['std'] / math.sqrt(r['count'])) if r['count'] > 1 else 0.0, axis=1)
        stats['ci95'] = 1.96 * stats['se']

        x_positions = np.arange(len(llms))
        y_max_data = sub['Accuracy'].max() if not sub['Accuracy'].isna().all() else 0.0
        y_min_data = sub['Accuracy'].min() if not sub['Accuracy'].isna().all() else 0.0
        span = max(y_max_data - y_min_data, 0.1)

        # Plot mean and 95% CI
        for xi, llm in zip(x_positions, llms):
            m = stats.loc[llm, 'mean']
            ci = stats.loc[llm, 'ci95']
            n = int(stats.loc[llm, 'count'])
            if n > 0 and not np.isnan(m):
                ax.errorbar(
                    x=xi, y=m, yerr=ci if ci > 0 else 0.01,
                    fmt='D', markersize=7, elinewidth=2, capsize=5,
                    ecolor='black',
                    markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.2,
                    zorder=15
                )
                ax.text(
                    xi, m + max(ci, 0.01) * 0.6, f"{m:.3f}",
                    ha='center', va='bottom', fontsize=9, zorder=20
                )

        # X-axis labels with n
        xtick_labels = [f"{llm}\n(n={int(stats.loc[llm,'count'])})" for llm in llms]
        ax.set_xticks(x_positions)
        rot = 25 if len(llms) > 3 else 0
        ax.set_xticklabels(xtick_labels, rotation=rot, ha='center')

        # Tukey HSD significance annotations
        sig_pairs = []
        try:
            nonzero = stats[stats['count'] > 0].index.tolist()
            if len(nonzero) >= 2:
                tukey = pairwise_tukeyhsd(endog=sub["Accuracy"], groups=sub["LLM"], alpha=0.05)
                for row in tukey.summary().data[1:]:
                    g1, g2, meandiff, p_adj, lower, upper, reject = row
                    if reject:
                        sig_pairs.append((g1, g2, float(p_adj)))
        except Exception as e:
            print(f"[WARN] Tukey failed on Detector={det}: {e}")

        if sig_pairs:
            base = y_max_data + 0.06 * span
            step = 0.06 * span
            used = {}
            for i, (g1, g2, pval) in enumerate(sig_pairs):
                if g1 not in llms or g2 not in llms:
                    continue
                x1 = llms.index(g1)
                x2 = llms.index(g2)
                key = tuple(sorted((x1, x2)))
                level = used.get(key, 0)
                used[key] = level + 1
                y = base + level * step
                # Draw significance line
                ax.plot([x1, x1, x2, x2], [y-0.002, y, y, y-0.002],
                        lw=1.5, color='black', zorder=25)
                # Format p-value
                txt = 'p<0.001' if pval < 0.001 else f"p={pval:.3f}"
                ax.text((x1+x2)/2, y + 0.01*span, txt, ha='center', va='bottom', fontsize=10, zorder=30)
            ax.set_ylim(top=base + (max(used.values()) if used else 0) * step + 0.05*span)

        ax.set_title(f"Detector: {det}")

    fig.supylabel('Accuracy')
    plt.subplots_adjust(bottom=0.25, wspace=0.3)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()



# ---------- Heatmap: mean accuracy by LLM x Detector ----------
def plot_heatmap_llm_detector(data, outpath='fig_heatmap_accuracy.png'):
    pivot = data.pivot_table(index="LLM", columns="Detector", values="Accuracy", aggfunc="mean")
    plt.figure(figsize=(5,3.6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Mean Accuracy'})
    plt.title('Mean Accuracy by LLM and Detector')
    plt.ylabel('LLM generator')
    plt.xlabel('Detector')
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


plot_cta_with_stats(data, outpath='fig_cta_accuracy.png')
plot_heatmap_llm_detector(data, outpath='fig_heatmap_accuracy.png')
plot_by_llm_detector_boxes(data, outpath='fig_llm_detector_boxes.png')


import math
from statsmodels.stats.multicomp import pairwise_tukeyhsd

cta_order = ['E', 'S', 'H']
stats_list = []

for cta in cta_order:
    subset = data[data['cta'] == cta]['Accuracy'].dropna()
    n = len(subset)
    mean = subset.mean()
    std = subset.std(ddof=1)
    se = std / math.sqrt(n) if n > 0 else np.nan
    ci95 = 1.96 * se
    stats_list.append({
        'CTA': cta,
        'n': n,
        'mean': mean,
        'std': std,
        'se': se,
        'ci95': ci95
    })

df_stats = pd.DataFrame(stats_list)
print(df_stats)

# Tukey HSD
tukey = pairwise_tukeyhsd(endog=data["Accuracy"], groups=data["cta"], alpha=0.05)
print(tukey.summary())

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:1080'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:1080'
# 设置代理地址
proxies = {
    "http": "http://127.0.0.1:1080",
    "https": "http://127.0.0.1:1080"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'ElSlay/BERT-Phishing-Email-Model'
# ---------------- 配置 ----------------

csv_files = [
    #("../../fusion_dataset_output/phishing_fused_1000.csv", "orig_pred", "../../models_output/bert/orig/first_phishing_fused_1000_predictions.csv"),  # 原始前1000条
    ("../../gpt_dataset_output/300/gpt_fused_phishing_output.csv", "gpt_pred",
     "../../models_output/bert/gpt/300/gpt_phishing_fused_predictions.csv")  # GPT改写后的
    #("../../qwen3_dataset_output/300/qwen3_fused_phishing_output_l.csv", "qwen3_pred",
    # "../../models_output/bert/qwen3/300/qwen3_phishing_fused_predictions_l.csv")  # GPT改写后的
]
# 自动检测设备：如果CUDA可用则使用GPU，否则使用CPU
model = BertForSequenceClassification.from_pretrained(
    model_name,
    proxies=proxies
)
tokenizer = BertTokenizer.from_pretrained(
    model_name,
    proxies=proxies
)
model.eval()
model.to(device)


# ---------------- 批量推理函数 ----------------
# def predict_batch(text_list, batch_size=8):
#     all_preds = []
#     for i in range(0, len(text_list), batch_size):
#         batch_texts = text_list[i:i + batch_size]
#         inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
#         all_preds.extend(preds)
#     return all_preds


from tqdm import tqdm

def predict_batch(text_list, batch_size=8):
    all_preds = []
    for i in tqdm(range(0, len(text_list), batch_size), desc="Predicting batches"):
        batch_texts = text_list[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
    return all_preds



# ---------------- 计算指标函数 ----------------
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    cm = confusion_matrix(y_true, y_pred)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "confusion_matrix": cm}


# ---------------- 对每个 CSV 执行 ----------------
for csv_path, pred_col, out_file in csv_files:
    print(f"\nProcessing {csv_path} ...")
    df = pd.read_csv(csv_path, dtype=str)

    if "label" not in df.columns:
        raise RuntimeError(f"{csv_path} 中没有 'label' 列，无法计算指标")

    # 批量预测
    df[pred_col] = predict_batch(df["text"].tolist())

    # 计算指标
    metrics = compute_metrics(df["label"].astype(int), df[pred_col])
    print(f"=== Metrics for {csv_path} ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])

    # 保存每条预测结果
    df.to_csv(out_file, index=False, encoding="utf-8")
    print(f"预测结果已保存到 {out_file}")

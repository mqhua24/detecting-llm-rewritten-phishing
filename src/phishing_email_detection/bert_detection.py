import pandas as pd
import torch


print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")

#'ElSlay/BERT-Phishing-Email-Model' 使用
from transformers import BertForSequenceClassification, BertTokenizer

#"ealvaradob/bert-finetuned-phishing" 使用
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
import os
from datetime import datetime

def detect_phishing_emails(input_csv, output_csv, text_column="text", label_column="label", batch_size=16, max_length=256):
    """
    使用BERT模型检测钓鱼邮件

    Args:
        input_csv (str): 输入CSV文件路径
        output_csv (str): 输出CSV文件路径
        text_column (str): 文本列名，默认为"text"
        label_column (str): 标签列名，默认为"label"
        batch_size (int): 批处理大小，默认为16
        max_length (int): 最大序列长度，默认为256

    Returns:
        dict: 包含评估指标的字典
    """

    # 检查CUDA是否可用，并设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # 设置代理地址
    proxies = {
        "http": "http://127.0.0.1:1080",
        "https": "http://127.0.0.1:1080"
    }

    # -----------------------------
    # 加载模型和 tokenizer
    # -----------------------------
    print("Loading BERT model...")


    # model_name = "../../models/bert-finetuned-phishing"
    #
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForSequenceClassification.from_pretrained(model_name)

    model_name = 'ElSlay/BERT-Phishing-Email-Model'

    # Load the pre-trained model and tokenizer
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

    # -----------------------------
    # 读取数据
    # -----------------------------
    df = pd.read_csv(input_csv)
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()

    # -----------------------------
    # 批量推理函数
    # -----------------------------
    def predict(texts, batch_size=16):
        all_preds = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k:v.to(device) for k,v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
        return all_preds

    # -----------------------------
    # 推理
    # -----------------------------
    print("Running predictions...")
    preds = predict(texts, batch_size=batch_size)
    print(preds)

    # -----------------------------
    # 计算指标
    # -----------------------------
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds)

    # -----------------------------
    # 输出结果
    # -----------------------------
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

    # -----------------------------
    # 保存预测结果
    # -----------------------------
    df["pred_label"] = preds
    df.to_csv(output_csv, index=False)
    print("Predictions saved to CSV.")

    # -----------------------------
    # 保存总体结果到单独文件（追加模式）
    # -----------------------------
    # 获取输出目录
    output_dir = os.path.dirname(output_csv)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 构造总体结果文件路径
    summary_file = os.path.join(output_dir, "detection_summary.txt")

    # 获取输入文件名（用于标识）
    input_filename = os.path.basename(input_csv)

    # 追加写入总体结果
    with open(summary_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input File: {input_filename}\n")
        f.write(f"Output File: {os.path.basename(output_csv)}\n")
        f.write(f"{'='*50}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
        f.write(f"Classification Report:\n{report}\n")
        f.write(f"{'='*50}\n\n")

    print(f"Summary results appended to {summary_file}")

    # 返回评估指标
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": report
    }

# 示例用法
if __name__ == "__main__":
    # 可以在这里调用函数
    #results = detect_phishing_emails("../../test_dataset_output/test_dataset.csv", "../../models_output/bert/test_dataset_pred.csv")
    pass

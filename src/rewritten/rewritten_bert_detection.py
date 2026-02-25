import os
import glob
from src.phishing_email_detection.bert_detection import detect_phishing_emails
from src.rewritten.rewritten_mail import input_path

import os
import glob
import pandas as pd

# 确定项目主目录
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_file_dir))

def add_label_column_to_csv_files(input_path):
    """
    为 rewritten_data 目录中 local-qwen3-4b-q8 子目录下的所有 CSV 文件添加 label 列（值为 1）
    """
    # 构建输入目录路径
    input_dir = os.path.join(project_root, 'rewritten_data', input_path)
    print(input_dir)
    # 查找所有 CSV 文件
    csv_pattern = os.path.join(input_dir, '*.csv')
    csv_files = glob.glob(csv_pattern)
    print(csv_files)
    # 处理每个 CSV 文件
    for csv_file in csv_files:
        try:
            # 读取 CSV 文件
            df = pd.read_csv(csv_file)

            # 添加 label 列，值都为 1
            df['label'] = 1

            # 保存修改后的文件（覆盖原文件）
            df.to_csv(csv_file, index=False)

            print(f"已为 {os.path.basename(csv_file)} 添加 label 列")

        except Exception as e:
            print(f"处理 {csv_file} 时出错: {e}")
            continue

def process_rewritten_emails(input_path):
    """
    处理rewritten_data目录中local-qwen3-4b-q8子目录下的所有CSV文件
    """

    # 构建输入目录路径
    # input_dir = os.path.join(project_root, 'rewritten_data', input_path)
    # input_dir = os.path.join(project_root, 'gpt_dataset_output', input_path)
    # input_dir = os.path.join(project_root, 'qwen3_dataset_output', input_path)
    #input_dir = os.path.join(project_root, 'together_llama_dataset_output', input_path)
    input_dir = os.path.join(project_root, input_path)

    # 构建输出目录路径(对应改变）
 #  output_dir = os.path.join(project_root, 'models_output', 'bert', '2000landp', input_path+'_gpt_ElSaly')
 #    output_dir = os.path.join(project_root, 'models_output', 'bert', 'rewritten', input_path + '_ealvaradob')
 #    output_dir = os.path.join(project_root, 'models_output', 'bert', '2000landp', input_path+'_llama_ElSaly')
 #    output_dir = os.path.join(project_root, 'models_output', 'bert', 'llama', 'local_1000', input_path + '_ElSaly')
    output_dir = os.path.join(project_root, 'models_output', 'cta_gate_bert_2', '2000landp', input_path+'_ElSaly')
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 查找所有CSV文件
    csv_pattern = os.path.join(input_dir, '*.csv')
    csv_files = glob.glob(csv_pattern)

    # 处理每个CSV文件
    for csv_file in csv_files:
        try:
            # 获取文件名（不含路径）
            filename = os.path.basename(csv_file)

            # 构造输出文件路径
            output_filename = filename.replace('.csv', '_pred.csv')
            output_file = os.path.join(output_dir, output_filename)

            print(f"Processing {filename}...")

            # 调用检测函数
            results = detect_phishing_emails(
                input_csv=csv_file,
                output_csv=output_file,
                text_column="result",  # 根据实际数据结构调整
                #text_column="text",
                label_column="label"   # 根据实际数据结构调整
            )

            print(f"Completed processing {filename}")
            print(f"Accuracy: {results['accuracy']:.4f}")
            print("-" * 50)

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue

if __name__ == "__main__":
    #增加label列
    #add_label_column_to_csv_files('local-qwen3-4b-q8')
   #add_label_column_to_csv_files('qwen3-30b-ali-format')
   # add_label_column_to_csv_files('together_llama_dataset_output')
    #检测处理邮件
     #process_rewritten_emails('local-qwen3-4b-q8')
   #process_rewritten_emails('qwen3-30b-ali-format')
    #process_rewritten_emails('2000legitandphish')
    # process_rewritten_emails('together_llama_dataset_output')
    #process_rewritten_emails('local_llama_dataset_output/llama3/2000legitandphish')
    # process_rewritten_emails('llm_cta_data/cta_rewritten/local-qwen3-4b-q8/concat2')
    # process_rewritten_emails('llm_cta_data/cta_rewritten/local-llama3-8b/concat2')
    process_rewritten_emails('llm_gate_data/local-llama3.2-11b/concat2')
import asyncio
from typing import List, Dict

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from src.llm.model_factory import ModelFactory
import os
import time

#model='qwen3-30b-ali'
#model='llmma-70b-bd'

#model='llmma-70b-to'
#model='local-gpt-oss-20b'

# model='local-llama3.2-11b'
# model='gpt-5-mini'
# model="local-qwen3-14b"
model="local-qwen3-4b-fp16"

current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_file_dir))

rewritten_path = os.path.join(project_root, 'llm_detect_data')

input_path = os.path.join(project_root, 'gpt_dataset_output', '2000legitandphish')
input_file = 'merged_shuffled_gpt_fused_with_id.csv'
output_dir = os.path.join(rewritten_path, model)
output_file = 'merged_shuffled_gpt_fused_with_id_detect_result.csv'

# input_file = 'merged_shuffled_fused_original_2000.csv'
# output_dir = os.path.join(rewritten_path, model)
# output_file = 'merged_shuffled_fused_original_2000_detect_result.csv'

# input_path = os.path.join(project_root, 'qwen3_dataset_output', '2000legitandphish')
# input_file = 'merged_shuffled_fused_original_2000_with_id.csv'
# output_dir = os.path.join(rewritten_path, model)
# output_file = 'merged_shuffled_fused_original_2000_with_id_detect_result.csv'

# input_path = os.path.join(project_root, 'qwen3_dataset_output', '2000legitandphish')
# input_file = 'merged_shuffled_qwen3_fused_m_clean_with_id.csv'
# output_dir = os.path.join(rewritten_path, model)
# output_file = 'merged_shuffled_qwen3_fused_m_clean_with_id_detect_result.csv'

# input_path = os.path.join(project_root, 'together_llama_dataset_output', '2000legitandphish')
# input_file = 'merged_shuffled_together_llama_fused_phishing_output.csv'
# output_dir = os.path.join(rewritten_path, model)
# output_file = 'merged_shuffled_together_llama_fused_phishing_output_detect_result.csv'

#
# input_path = os.path.join(project_root, 'local_llama_dataset_output', 'llama3', '2000legitandphish')
# input_file = 'merged_shuffled_local_llama_fused_phishing_output_cleaned_with_id.csv'
# output_dir = os.path.join(rewritten_path, model)
# output_file = 'merged_shuffled_local_llama_fused_phishing_output_detect_result.csv'

# 检查并创建model目录（如果不存在）Check and create the model directory (if it does not exist)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# def get_config_file(filename):
#     """获取配置文件完整路径"""
#     return os.path.join(rewritten_path, 'config', filename)
#
# system_prompt_file = get_config_file('system_prompt.md')
# human_prompt_file = get_config_file('human_prompt.md')
# factor_df_file = get_config_file('factor_table.csv')

def load_factor_combinations(csv_file):
    """
    从CSV文件加载因子组合
    Load the factor combination from the CSV file
    """
    return pd.read_csv(csv_file)

def create_chat_model():
    """
    创建聊天模型实例
    Create an instance of the chat model
    """
    config_path = ModelFactory.get_project_config_path()
    factory = ModelFactory(config_path)

    return factory.create_model(model)

def get_prompt_template(prompt_file_path):
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        # 读取文件内容并去除标题行 Read the file content and remove the title line
        content = f.read()
        # 提取实际提示词内容（跳过第一行标题）Extract the actual prompt content (skip the title of the first line)
        lines = content.strip().split('\n')
        return '\n'.join(lines[2:])  # 跳过标题和空行 Skip the title and blank lines

def create_prompt_template():
    """
    创建提示词模板
    Create a prompt word template
    """
    system_template = """
You are an email security classifier.
Your task is to decide whether an email is phishing or benign.

Output your answer in valid JSON with the following fields:
"label": 1 if the email is phishing, 0 if the email is benign.
"confidence": a number between 0 and 1
"reason": one short sentence explaining your decision

Output JSON only, with no extra text.
"""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = "{mail}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    return ChatPromptTemplate.from_messages([
        system_message_prompt,
        human_message_prompt
    ])

def extract_model_response(response) -> str:
    """
    统一提取不同模型的回复内容 Uniformly extract the response contents of different models

    Args:
        response: 模型响应对象 Model response object

    Returns:
        str: 模型回复的文本内容 The text content of the model's response
    """
    # 检查是否有 content 属性（大多数模型）Check if there is a content attribute (in most models)
    if hasattr(response, 'content') and response.content:
        return response.content
    # 检查是否有 content 属性（大多数模型）
    if hasattr(response, 'result') and response.result:
        return response.result

    # 对于 Ollama 可能直接返回字符串的情况 For the case where Ollama might directly return a string
    if isinstance(response, str):
        return response

    # 如果以上都不匹配，转换为字符串 If none of the above matches, convert to a string
    return str(response)

async def process_emails_with_factors(chat_prompt, chat_model, original_email, batch_size: int = 10):

    results = [None] * len(original_email)  # 保持原始顺序
    semaphore = asyncio.Semaphore(batch_size)
    # 分批处理邮件 Process emails in batches
    for i in range(0, len(original_email), batch_size):
        batch = original_email[i:i + batch_size]
        tasks = []

        async def limited_task(email, idx):
            async with semaphore:
                return await process_email_with_factors_async(
                chat_prompt,
                chat_model,
                email['content'],
                email['id']
            )

        # 为每封邮件创建异步任务 Create asynchronous tasks for each email
        for j, email in enumerate(batch):
            #print(j,len(email['content']))
            task = limited_task(email, j)
            tasks.append(task)

        # 并发执行当前批次的任务 Execute the tasks of the current batch concurrently
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        # print(batch_results)
        # 将结果按原始顺序存储 Store the results in their original order
        for j, result in enumerate(batch_results):
            original_index = i + j
            if isinstance(result, Exception):
                print(f"An exception occurred when handling the email {original_email[original_index]['id']} : {result}")
                #results[original_index] = None
                results[original_index] = {
                    "id": original_email[original_index]["id"],
                    "result": None,
                    "error": str(result)
                }
            else:
                results[original_index] = result

    return results


async def process_email_with_factors_async(chat_prompt, chat_model, original_email, original_email_id):
    """
    使用特定因子组合异步处理邮件
    Use specific factor combinations to process emails asynchronously
    """

    # 格式化提示词
    formatted_messages = chat_prompt.format_prompt(
        mail=original_email
    ).to_messages()

    start_time = time.time()
    print("Dealing with the email:",start_time ,original_email_id)
    #print(formatted_messages)
    # 异步调用模型获取响应 The asynchronous call model retrieves the response
    # 使用 asyncio.to_thread 将同步调用放到线程中执行 Use asyncio.to_thread to execute the synchronous call in the thread
    response = await asyncio.to_thread(chat_model.invoke, formatted_messages)
    response_content = extract_model_response(response)

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"One-time processing time: {processing_time:.2f} seconds")

    # 输出结果（可根据需要保存到文件） Output result (can be saved to a file as needed)
    print(f"Model response: {response_content}")
    print("=" * 50)

    # 返回包含所有信息的字典
    return {

        'result': response_content
    }

def read_emails_from_csv(filename: str) -> List[Dict]:
    """
    从CSV文件中读取邮件内容
    Read the email content from the CSV file

    Args:
        filename (str): CSV文件名

    Returns:
        List[Dict]: 邮件列表，每个元素包含邮件ID和内容 The mailing list, each element contains the email ID and content
    """
    emails = []
    try:
        df = pd.read_csv(filename)
        for index, row in df.iterrows():
            # 假设CSV文件中有'id'和'content'列，如果没有可以根据实际情况调整
            # Suppose there are columns 'id' and 'content' in the CSV file. If not, they can be adjusted according to the actual situation
            email_id = row.get('id', index + 1)
            email_content = row.get('text', str(row))
            emails.append({
                'id': email_id,
                'content': email_content
            })
    except Exception as e:
        print(f"Error occurred when reading the CSV file: {e}")
        return []

    return emails


async def main_async():
    # 加载因子组合 Combination of loading factors
    #factor_df = load_factor_combinations(factor_df_file)

    # 创建聊天模型和提示词模板 Create chat models and prompt word templates
    chat_model = create_chat_model()
    chat_prompt = create_prompt_template()
    original_file_path = os.path.join(input_path, input_file)
    # 原始邮件内容 Original email content
    original_email = read_emails_from_csv(original_file_path)

    # 提取基础文件名 Extract the basic file name
    base_filename = os.path.splitext(os.path.basename(original_file_path))[0]



    output_csv_file = os.path.join(output_dir, output_file)

    # 使用当前因子组合处理邮件 Process emails using the current factor combination
    result = await process_emails_with_factors(
        chat_prompt,
        chat_model,
        original_email
    )

    # 将结果保存到CSV文件 Save the result to a CSV file
    results_df = pd.DataFrame(result)
    results_df.to_csv(output_csv_file, index=False, encoding='utf-8')
    print(f"The result has been saved to {output_csv_file} ")

if __name__ == "__main__":

    start_time = time.time()
    asyncio.run(main_async())
    # 计算处理时间 Calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Processing time: {processing_time:.2f} seconds")
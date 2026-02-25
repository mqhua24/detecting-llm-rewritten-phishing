import asyncio

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from style_mappings import get_style_description as get_style
from src.llm.model_factory import ModelFactory
import os
import time

def get_config_file(filename):
    """获取配置文件完整路径"""
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_file_dir))
    return os.path.join(project_root, 'rewritten_data', 'config', filename)

system_prompt_file = get_config_file('system_prompt.md')
human_prompt_file = get_config_file('human_prompt.md')
factor_df_file = get_config_file('factor_table.csv')

def load_factor_combinations(csv_file):
    """
    从CSV文件加载因子组合
    """
    return pd.read_csv(csv_file)

def create_chat_model():
    """
    创建聊天模型实例
    """
    config_path = ModelFactory.get_project_config_path()
    factory = ModelFactory(config_path)

    return factory.create_model('local-qwen3-4b-q8')

def get_prompt_template(prompt_file_path):
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        # 读取文件内容并去除标题行
        content = f.read()
        # 提取实际提示词内容（跳过第一行标题）
        lines = content.strip().split('\n')
        return '\n'.join(lines[2:])  # 跳过标题和空行

def create_prompt_template():
    """
    创建提示词模板
    """
    system_template = get_prompt_template(system_prompt_file)
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = get_prompt_template(human_prompt_file)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    return ChatPromptTemplate.from_messages([
        system_message_prompt,
        human_message_prompt
    ])

def extract_model_response(response) -> str:
    """
    统一提取不同模型的回复内容

    Args:
        response: 模型响应对象

    Returns:
        str: 模型回复的文本内容
    """
    # 检查是否有 content 属性（大多数模型）
    if hasattr(response, 'content') and response.content:
        return response.content

    # 对于 Ollama 可能直接返回字符串的情况
    if isinstance(response, str):
        return response

    # 如果以上都不匹配，转换为字符串
    return str(response)


async def process_email_with_factors_async(chat_prompt, chat_model, original_email, combo_id, cta, amount, sensitive):
    """
    使用特定因子组合异步处理邮件
    """
    # 获取各因子的描述文本
    cta_desc = get_style("CTA", cta)
    amount_desc = get_style("AMOUNT", amount)
    sensitive_desc = get_style("SENSITIVE", sensitive)

    # 格式化提示词
    formatted_messages = chat_prompt.format_prompt(
        ORIG_EMAIL_TEXT=original_email,
        CTA_STYLE=cta_desc,
        AMOUNT_STYLE=amount_desc,
        SENSITIVE_STYLE=sensitive_desc,
    ).to_messages()

    start_time = time.time()

    # 异步调用模型获取响应
    # 使用 asyncio.to_thread 将同步调用放到线程中执行
    response = await asyncio.to_thread(chat_model.invoke, formatted_messages)
    response_content = extract_model_response(response)

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"一次处理时间: {processing_time:.2f} 秒")

    # 输出结果（可根据需要保存到文件）
    print(f"=== 组合 {combo_id} 结果 ===")
    print(f"CTA: {cta}, Amount: {amount}, Sensitive: {sensitive}")
    print(f"模型响应: {response_content}")
    print("=" * 50)

    # 返回包含所有信息的字典
    return {
        'combo_id': combo_id,
        'cta': cta,
        'amount': amount,
        'sensitive': sensitive,
        'result': response_content
    }

# !/usr/bin/env python3
import re
from typing import Dict, List, Optional, Set
import pandas as pd


def read_spam_results(filename: str = "spam_results.csv") -> list:
    """
    读取spam结果CSV文件，返回Response列的值列表
    Read the CSV file of spam results and return the list of values in the "Response" column

    Args:
        filename (str): CSV文件名，默认为"spam_results.csv"  CSV file name, the default is "spam_results.csv"

    Returns:
        list: Response列的值列表  The list of values in the Response column
    """
    try:
        # 读取CSV文件 Read the CSV file
        df = pd.read_csv(filename)

        # 提取Response列并转换为列表 Extract the "Response" column and convert it into a list
        response_list = df['Response'].tolist()

        return response_list
    except Exception as e:
        print(f"Error occurred while reading the file: {e}")
        return []
"""
解析SpamAssassin响应并生成CSV文件 
Parse the SpamAssassin response and generate a CSV file
"""

def parse_spamassassin_response(response_text: str, email_id: int) -> Dict:
    """
    解析SpamAssassin响应文本，提取关键信息
    Analyze the response text of SpamAssassin and extract the key information

    Args:
        response_text (str): SpamAssassin的响应文本 The response text of SpamAssassin
        email_id (int): 邮件ID Email ID

    Returns:
        Dict: 包含解析结果的字典 A dictionary containing the parsing results
    """
    result = {"Email": email_id}

    # 提取Spam状态 Extract the Spam status
    spam_match = re.search(r'Spam:\s+(True|False)', response_text)
    if spam_match:
        result["Spam"] = spam_match.group(1)

    # 提取分数 extraction score
    score_match = re.search(r'Spam:\s+\w+\s*;\s*([0-9.]+)\s*/\s*([0-9.]+)', response_text)
    if score_match:
        result["Score"] = float(score_match.group(1))
        result["Required"] = float(score_match.group(2))

    # 动态提取所有规则名称和分数 Dynamically extract all rule names and scores
    lines = response_text.split('\n')
    in_analysis = False

    for line in lines:
        # 检测是否进入分析部分 Check whether it enters the analysis section
        if "Content analysis details" in line:
            in_analysis = True
            continue

        # 如果已经进入分析部分，且行符合规则格式 If you have already entered the analysis section and the rows conform to the rule format
        if in_analysis and re.match(r'^\s*-?[0-9.]+\s+', line):
            # 提取分数和规则名 Extract the score and rule name
            parts = line.split()
            if len(parts) >= 3:
                try:
                    score = float(parts[0])
                    rule_name = parts[1]
                    result[rule_name] = score
                except ValueError:
                    continue
        # 如果遇到明显的结束标记，则停止分析 If a clear end marker is encountered, stop the analysis
        elif in_analysis and ("Content preview:" in line):
            # 只有遇到明确的结束标记才停止 It only stops when a clear end sign is encountered
            in_analysis = False

    return result





def save_to_csv(results: List[Dict], filename: str = "spam_results.csv"):
    """
    使用pandas将解析结果保存为CSV文件
    Save the parsing result as a CSV file using pandas

    Args:
        results (List[Dict]): 解析结果列表 Parse the result list
        filename (str): 输出文件名 output file name
    """
    if not results:
        return

    # 转换为DataFrame Convert to DataFrame
    df = pd.DataFrame(results)

    # 保存为CSV文件 Save as a CSV file
    df.to_csv(filename, index=False, encoding='utf-8')


def collect_all_rule_names(responses: List[str]) -> Set[str]:
    """
    从所有响应中收集所有的规则名称
    Collect all the rule names from all responses

    Args:
        responses (List[str]): 所有响应文本列表 List of all response texts

    Returns:
        Set[str]: 所有唯一规则名称集合 A collection of all unique rule names
    """
    all_rules = set()
    for response in responses:
        lines = response.split('\n')
        in_analysis = False

        for line in lines:
            # 检测是否进入分析部分 Check whether it enters the analysis section
            if "Content analysis details" in line:
                in_analysis = True
                continue

            # 如果已经进入分析部分，且行符合规则格式
            # If you have already entered the analysis section and the rows conform to the rule format
            if in_analysis and re.match(r'^\s*-?[0-9.]+\s+', line):
                # 提取分数和规则名 Extract the score and rule name
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        float(parts[0])  # 验证是否为有效数字 Verify whether it is a significant figure
                        rule_name = parts[1]
                        all_rules.add(rule_name)
                    except ValueError:
                        continue
            # 如果遇到明显的结束标记，则停止分析 If a clear end marker is encountered, stop the analysis
            elif in_analysis and ("Content preview:" in line or line.strip() == "" and "----" in line):
                # 只有遇到明确的结束标记才停止，而不是任何Content开头的行
                # Stop only when a clear end tag is encountered, not any line starting with "Content"
                in_analysis = False

    return all_rules

def process_multiple_responses(responses: List[str]) -> List[Dict]:
    """
    处理多个SpamAssassin响应
    Handling multiple SpamAssassin responses

    Args:
        responses (List[str]): 响应文本列表 Response text list

    Returns:
        List[Dict]: 解析结果列表 Parse the result list
    """
    all_rules = collect_all_rule_names(responses)
    print(f"All the collected rules: {all_rules}")  # 调试信息 debugging information

    results = []
    for i, response in enumerate(responses, 1):
        parsed_result = parse_spamassassin_response(response, i)
        print(f"The rule in the email {i} parsing result: {[k for k in parsed_result.keys() if k not in ['Email', 'Spam', 'Score', 'Required']]}")  # 调试信息
        # 确保所有规则字段都存在于结果中 Make sure that all rule fields exist in the result
        for rule in all_rules:
            if rule not in parsed_result:
                parsed_result[rule] = 0.0
        results.append(parsed_result)
    return results

def spam_data_check(input_file, output_file ):
    sample_responses = read_spam_results(input_file)

    # 处理响应并生成CSV Process the response and generate a CSV
    results = process_multiple_responses(sample_responses)
    save_to_csv(results, output_file)

# 示例使用
if __name__ == "__main__":

    spam_data_check("spam_results.csv" , "spam_results_data.csv")

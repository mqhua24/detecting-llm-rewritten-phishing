#!/usr/bin/env python3
"""
SpamAssassin客户端 - 通过端口783连接spamd服务
支持批量处理和异步调用，直接存储完整响应结果
SpamAssassin Client - Connect to the spamd service via port 783
Supports batch processing and asynchronous calls, and directly stores the complete response results
"""

import asyncio
import pandas as pd
from typing import List, Dict, Optional
from spamassassin_data import spam_data_check

async def check_spamassassin_async(host: str, port: int, email_content: str, email_id: int, timeout: int = 60) -> Optional[Dict]:
    """
    异步方式通过端口783连接SpamAssassin的spamd服务进行垃圾邮件检测
    Connect to SpamAssassin's spamd service asynchronously via port 783 for spam detection

    Args:
        host (str): spamd服务器主机名
        port (int): spamd服务端口
        email_content (str): 要检测的邮件内容
        email_id (int): 邮件ID
        timeout (int): 连接超时时间，默认为30秒

        host (str): spamd server hostname
        port (int): spamd service port
        email_content (str): The content of the email to be detected
        email_id (int): Mail ID
        timeout (int): Connection timeout period, with a default of 30 seconds

    Returns:
        Optional[Dict]: 包含邮件ID和完整响应的字典，如果出错则返回None
        Optional[Dict]: A dictionary containing the email ID and the full response, which returns None if an error occurs
    """
    try:
        # 创建socket连接 Establish a socket connection
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout
        )

        # 构建SPAMD协议请求 Request for establishing the SPAMD protocol
        content_with_newline = email_content + "\r\n"
        content_bytes = content_with_newline.encode('utf-8')
        #print(f"Content字符长度: {len(email_content)}")
        #print(f"Content字节长度: {len(content_bytes)}")
        request_lines = [
            "REPORT SPAMC/1.2",
            f"Content-length: {len(content_bytes)}",
            "User: testuser",
            "",  # 分割头部和内容 Separate the header from the content
            content_with_newline
        ]
        request = "\r\n".join(request_lines)

        # 发送请求 Send Request
        writer.write(request.encode('utf-8'))
        await writer.drain()

        # 接收响应 receiving response
        response = b""
        while True:
            try:
                data = await asyncio.wait_for(reader.read(4096), timeout=timeout)
                if not data:
                    break
                response += data
            except asyncio.TimeoutError:
                break

        writer.close()
        await writer.wait_closed()

        # 解析响应 Parsing response
        response_text = response.decode('utf-8')
        print({
            "Email": email_id,
            "Response": response_text[:40]
        })
        # 返回完整响应结果 Return the complete response result
        return {
            "Email": email_id,
            "Response": response_text
        }

    except asyncio.TimeoutError:
        print(f"Connection timeout: Unable to connect to {host}:{port} within {timeout} seconds")
        return None
    except ConnectionRefusedError:
        print(f"Connection rejected: Please ensure that the spamd service is running at {host}:{port}")
        return None
    except Exception as e:
        print(f"Error occurred: {e}")
        return None


def read_emails_from_csv(filename: str) -> List[Dict]:
    """
    从CSV文件中读取邮件内容
    Read the email content from the CSV file

    Args:
        filename (str): CSV文件名  CSV file name

    Returns:
        List[Dict]: 邮件列表，每个元素包含邮件ID和内容  The mailing list, each element contains the email ID and content
    """
    emails = []
    try:
        df = pd.read_csv(filename)
        for index, row in df.iterrows():
            # 假设CSV文件中有'id'和'content'列，如果没有可以根据实际情况调整
            # Suppose there are columns 'id' and 'content' in the CSV file. If not, they can be adjusted according to the actual situationSuppose there are columns 'id' and 'content' in the CSV file. If not, they can be adjusted according to the actual situation
            email_id = row.get('id', index + 1)
            email_content = row.get('text', str(row))
            emails.append({
                'id': email_id,
                'content': email_content
            })
    except Exception as e:
        print(f"Error in reading CSV file: {e}")
        return []

    return emails


async def process_emails_batch(emails: List[Dict], host: str = 'localhost', port: int = 783,
                              batch_size: int = 10) -> List[Dict]:
    """
    批量异步处理邮件
    Batch asynchronous processing of emails

    Args:
        emails (List[Dict]): 邮件列表 Mailing list
        host (str): spamd服务器主机名 spamd server hostname
        port (int): spamd服务端口 spamd service port
        batch_size (int): 批处理大小 Batch size

    Returns:
        List[Dict]: 处理结果列表，保持原始顺序 List of processing results, maintaining the original order

    """
    results = [None] * len(emails)  # 保持原始顺序 Maintain the original order
    semaphore = asyncio.Semaphore(1)
    # 分批处理邮件 Process emails in batches
    for i in range(0, len(emails), batch_size):
        batch = emails[i:i + batch_size]
        tasks = []

        async def limited_task(email, idx):
            async with semaphore:
                return await check_spamassassin_async(host, port, email['content'], email['id'])

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
                print(f"An exception occurred when handling the email {emails[original_index]['id']} : {result}")
                results[original_index] = None
            else:
                results[original_index] = result

    return results


def save_results_to_csv(results: List[Dict], filename: str = "spam_results.csv"):
    """
    将处理结果保存到CSV文件 Save the processing result to a CSV file

    Args:
        results (List[Dict]): 处理结果列表 List of processing results
        filename (str): 输出文件名 outfilename
    """
    if not results:
        return

    # 过滤掉None结果
    valid_results = [r for r in results if r is not None]

    if not valid_results:
        return

    # 转换为DataFrame并保存
    df = pd.DataFrame(valid_results)
    df.to_csv(filename, index=False, encoding='utf-8-sig')


async def main():
    """
    主函数 - 从CSV文件读取邮件并批量处理
    The main function - reads emails from CSV files and processes them in batches
    """
    # 配置参数 configuration parameter
    # input_csv = "../../qwen3_dataset_output/1000/qwen3_fused_phishing_output_m_new.csv"  # 输入CSV文件名
    # output_csv = "qwen3_fused_phishing_output_spam_results_m_new.csv"  # 输出CSV文件名
    # output1_csv = "qwen3_fused_phishing_output_spam_results_data_m_new.csv"  # 输出CSV文件名

    # input_csv = "../../gpt_dataset_output/1000/gpt_fused_phishing_output.csv"  # 输入CSV文件名
    # output_csv = "gpt_fused_phishing_output_spam_results.csv"  # 输出CSV文件名
    # output1_csv = "gpt_fused_phishing_output_spam_results_data.csv"  # 输出CSV文件名

    # input_csv = "../../qwen3_dataset_output/1000/qwen3_fused_phishing_output_clean.csv"  # 输入CSV文件名
    # output_csv = "qwen3_fused_phishing_output_spam_results_clean.csv"  # 输出CSV文件名
    # output1_csv = "qwen3_fused_phishing_output_spam_results_data_clean.csv"  # 输出CSV文件名

    # input_csv = "../../qwen3_dataset_output/2000legitandphish/merged_shuffled_fused_original_2000.csv"  # 输入CSV文件名
    # output_csv = "2000/merged_shuffled_fused_original_2000_spam_results.csv"  # 输出CSV文件名
    # output1_csv = "2000/merged_shuffled_fused_original_2000_spam_results_data_clean.csv"  # 输出CSV文件名

    # 2000条qwen3数据
    # input_csv = "../../qwen3_dataset_output/2000legitandphish/merged_shuffled_qwen3_fused_m_clean.csv"  # 输入CSV文件名
    # output_csv = "2000/merged_shuffled_qwen3_fused_m_clean_2000_spam_results.csv"  # 输出CSV文件名
    # output1_csv = "2000/merged_shuffled_qwen3_fused_m_clean_2000_spam_results_data_clean.csv"  # 输出CSV文件名

    # 2000条gpt数据
    # input_csv = "../../gpt_dataset_output/2000legitandphish/merged_shuffled_gpt_fused.csv"  # 输入CSV文件名
    # output_csv = "2000/merged_shuffled_gpt_fused_2000_spam_results.csv"  # 输出CSV文件名
    # output1_csv = "2000/merged_shuffled_gpt_fused_2000_spam_results_data_clean.csv"  # 输出CSV文件名

    # 跑的 factor的数据27组
    # input_csv = "../../rewritten_data/local-qwen3-4b-q8/phishing_fused_1000_1_E_N_K.csv"  # 输入CSV文件名
    # output_csv = "../../rewritten_data/local-qwen3-4b-q8_spam_result/phishing_fused_1000_1_E_N_K_spam_results.csv"  # 输出CSV文件名
    # output1_csv = "../../rewritten_data/local-qwen3-4b-q8_spam_result/phishing_fused_1000_1_E_N_K_spam_results_data.csv"  # 输出CSV文件名

    # #2000条llama
    # input_csv = "../../together_llama_dataset_output/2000legitandphish/merged_shuffled_together_llama_fused_phishing_output_m.csv"  # 输入CSV文件名
    # output_csv = "2000/merged_shuffled_together_llama_fused_m_2000_spam_results.csv"  # 输出CSV文件名
    # output1_csv = "2000/merged_shuffled_together_llama_fused_m_2000_spam_results_data_clean.csv"  # 输出CSV文件名

    #1000条llama
    # input_csv = "../../together_llama_dataset_output/together_llama_fused_phishing_output.csv"  # 输入CSV文件名
    # output_csv = "together_llama_fused_phishing_output_spam_results.csv"  # 输出CSV文件名
    # output1_csv = "together_llama_fused_phishing_output_spam_results_data_clean.csv"  # 输出CSV文件名

    # input_csv = "../../local_llama_dataset_output/llama3/clean/local_llama_fused_phishing_output_l_new2_cleaned.csv"  # 输入CSV文件名
    # output_csv = "local_llama_fused_phishing_output_l_new2_spam_results.csv"  # 输出CSV文件名
    # output1_csv = "local_llama_fused_phishing_output_l_new2_spam_results_data_clean.csv"  # 输出CSV文件名

    input_csv = "../../local_llama_dataset_output/llama3/2000legitandphish/merged_shuffled_local_llama_fused_phishing_output_cleaned_with_id.csv"  # 输入CSV文件名
    output_csv = "2000/merged_shuffled_local_llama_fused_2000_spam_results.csv"  # 输出CSV文件名
    output1_csv = "2000/merged_shuffled_local_llama_fused_2000_spam_results_data_clean.csv"  # 输出CSV文件名

    host = '23.166.168.189'
    port = 783
    batch_size = 10  # 批处理大小 Batch size

    # 从CSV文件读取邮件 Read emails from CSV files
    print("Reading the email...")
    emails = read_emails_from_csv(input_csv)
    if not emails:
        print("The email was not read or the reading failed")
        return
    print(emails)
    print(f"Successfully read the {len(emails)} emails")

    # 批量异步处理邮件 Batch asynchronous processing of emails
    print("Start dealing with the emails...")
    results = await process_emails_batch(emails, host, port, batch_size)

    # 保存结果到CSV文件 Save the result to a CSV file
    print("The result is being saved...")
    save_results_to_csv(results, output_csv)
    print(f"The result has been saved to {output_csv}")

    spam_data_check(output_csv, output1_csv)


if __name__ == "__main__":
    # 运行异步主函数 Run the asynchronous main function
    asyncio.run(main())
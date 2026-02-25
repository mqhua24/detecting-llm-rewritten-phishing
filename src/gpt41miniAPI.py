import os
import time
import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import tiktoken  # ✅ 新增：计算 token 数
from prompt import prompt_prefix, prompt_prefix_m, prompt_prefix_l

# ----------------- 配置项 -----------------
type = "_l"
INPUT_CSV = "../fusion_dataset_output/phishing_fused.csv"
OUTPUT_CSV = f"../gpt_dataset_output/gpt_fused_phishing_output{type}.csv"
ERROR_CSV = f"../gpt_dataset_output/gpt_fused_phishing_errors{type}.csv"
PROXY = "http://127.0.0.1:1080"
MODEL = "gpt-4.1-mini"
MAX_WORKERS = 5
SLEEP_BETWEEN_CALLS = 0.0
TIMEOUT = 20.0
MAX_RETRIES = 3
TOKEN_LIMIT = 300000  # ✅ 超过这个 token 数就跳过
# ------------------------------------------

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("请先设置环境变量 OPENAI_API_KEY")

transport = httpx.HTTPTransport(proxy=PROXY)
http_client = httpx.Client(transport=transport, timeout=TIMEOUT)

client = OpenAI(
    api_key=api_key,
    http_client=http_client,
    timeout=TIMEOUT,
    max_retries=MAX_RETRIES
)

# 初始化 tokenizer（根据模型选择编码器）
enc = tiktoken.encoding_for_model(MODEL)

if type == "":
    prompt_prefix_selected = prompt_prefix
elif type == "_l":
    prompt_prefix_selected = prompt_prefix_l
elif type == "_m":
    prompt_prefix_selected = prompt_prefix_m


# -------------------- 核心函数 --------------------

@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_api_single(content: str):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"{prompt_prefix_selected}{content}"}]
    )
    try:
        return response.choices[0].message.content
    except Exception:
        try:
            return response.choices[0].text
        except Exception:
            raise


def safe_call(idx: int, orig_text: str):
    """安全调用，带 token 检查与异常捕获"""
    try:
        prompt = f"{prompt_prefix_selected}{orig_text}"
        token_count = len(enc.encode(prompt))

        if token_count > TOKEN_LIMIT:
            msg = f"Too long – skipped ({token_count} tokens)"
            print(f"[Skip] row {idx} 太长 ({token_count} tokens)，跳过。")
            return orig_text, RuntimeError(msg)

        rewritten = call_api_single(orig_text)
        if rewritten is None:
            return None, RuntimeError("API 返回空")
        return rewritten.strip(), None
    except RetryError as re:
        last_exc = re.last_attempt.exception() if hasattr(re, "last_attempt") else re
        return None, last_exc
    except Exception as e:
        return None, e


def batch_process_concurrent(input_csv: str, output_csv: str, error_csv: str,
                             max_workers: int = 5, sleep_between: float = 0.0, test_n: int = None):
    df = pd.read_csv(input_csv, dtype=str)
    if "text" not in df.columns:
        raise RuntimeError("输入 CSV 中找不到 'text' 列，请检查文件。")

    if test_n is not None:
        df = df.head(test_n)

    results = []
    errors = []

    # 创建一个索引到结果位置的映射
    index_mapping = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for idx, row in df.iterrows():
            orig_text = row.get("text", "") or ""
            future = executor.submit(safe_call, idx, orig_text)
            future_to_idx[future] = (idx, orig_text)
            index_mapping[idx] = len(results)
            results.append(None)  # 预留位置

        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Processing rows"):
            idx, orig_text = future_to_idx[future]
            try:
                rewritten, err = future.result()
            except Exception as e:
                rewritten, err = None, e

            # 按原始索引位置存储结果
            result_entry = {"text": orig_text, "label": 1}
            if not err:
                result_entry = {"text": rewritten, "label": 1}
            else:
                errors.append({"index": idx, "text": orig_text, "error": repr(err)})

            results[index_mapping[idx]] = result_entry

            if sleep_between:
                time.sleep(sleep_between)

    out_df = pd.DataFrame(results, columns=["text", "label"])
    out_df.to_csv(output_csv, index=False)
    print(f"处理完成，输出写入 {output_csv}，共 {len(out_df)} 条。")

    if errors:
        err_df = pd.DataFrame(errors)
        err_df.to_csv(error_csv, index=False)
        print(f"{len(errors)} 条调用失败或被跳过，详情写入 {error_csv}")

# -------------------- 主入口 --------------------

if __name__ == "__main__":
    TEST_N = 1000  # 调试时改成 1 或 5；正式跑可以设为 None
    batch_process_concurrent(INPUT_CSV, OUTPUT_CSV, ERROR_CSV,
                             max_workers=MAX_WORKERS,
                             sleep_between=SLEEP_BETWEEN_CALLS,
                             test_n=TEST_N)

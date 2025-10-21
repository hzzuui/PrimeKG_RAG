import json
import requests
import os
from datetime import datetime

# === 檔案路徑 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(os.path.dirname(BASE_DIR), "data", "primekg_queries.jsonl")
OUTPUT_PATH = os.path.join(os.path.dirname(BASE_DIR), "data", "primekg_queries_expanded.jsonl")
LOG_PATH = os.path.join(os.path.dirname(BASE_DIR), "data", "expansion_log.txt")

# === 呼叫 LLM 的函式 ===
import re

# === 呼叫 LLM 的函式（改良版） ===
def expand_with_llm(term, model="qwen3:14b"):
    """
    使用 LLM 將名詞翻譯成中文，並生成同義詞或縮寫。
    自動處理：若 LLM 回傳非 JSON 格式，會嘗試正則擷取 JSON。
    """
    prompt = f"""
請針對以下醫學名詞，輸出繁體中文翻譯，以及 2–3 個常見同義詞或縮寫（如果有）。
名詞: {term}

請用 JSON 格式回覆，不能有多餘文字，例如：
{{
  "chinese": "中文翻譯",
  "variants": ["同義詞1", "縮寫2"]
}}
"""

    try:
        resp = requests.post(
            "http://127.0.0.1:5000/rag",
            json={"query": prompt, "mode": "不使用 RAG 回應", "model": model}
        )
        if resp.status_code == 200:
            ans = resp.json().get("answer_no_rag", "{}")
            print(f"[DEBUG] LLM 回應原始內容: {ans}")  # <-- Debug 用

            # 嘗試直接解析
            try:
                return json.loads(ans)
            except:
                # 嘗試正則擷取 JSON 區塊
                match = re.search(r"\{.*\}", ans, re.S)
                if match:
                    try:
                        return json.loads(match.group())
                    except Exception as e:
                        print(f"[WARN] JSON parse error after regex: {e}")
                # fallback → 原文
                return {"chinese": term, "variants": []}
        else:
            print(f"[ERROR] 翻譯失敗: {resp.text}")
            return {"chinese": term, "variants": []}
    except Exception as e:
        print(f"[ERROR] LLM 請求錯誤: {e}")
        return {"chinese": term, "variants": []}

# === 主程式 ===
def expand_gold_answers(
    input_path=INPUT_PATH,
    output_path=OUTPUT_PATH,
    log_path=LOG_PATH,
    model="qwen3:14b",
    only_multihop=True
):
    """
    請針對以下醫學名詞，輸出繁體中文翻譯，以及 2–3 個常見同義詞或縮寫（如果有）。
    名詞: {term}

    請用 JSON 格式回覆，不能有額外文字，例如：
    {{
    "chinese": "中文翻譯",
    "variants": ["同義詞1", "縮寫2"]
    }}
    """

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout, \
         open(log_path, "a", encoding="utf-8") as flog:

        flog.write(f"\n===== Expansion Log {datetime.now()} =====\n")

        for line in fin:
            q = json.loads(line)

            # ✅ 過濾題型
            if only_multihop and q.get("type") != "multi-hop":
                fout.write(json.dumps(q, ensure_ascii=False) + "\n")
                continue

            expanded_answers = []
            for ans in q["gold_answers"]:
                expanded_answers.append(ans)  # 原始英文
                res = expand_with_llm(ans, model=model)

                # 寫 log
                flog.write(f"\n[原始]: {ans}\n")
                flog.write(f"[中文]: {res.get('chinese','')}\n")
                flog.write(f"[變體]: {', '.join(res.get('variants', []))}\n")

                if res["chinese"]:
                    expanded_answers.append(res["chinese"])  # 中文翻譯
                expanded_answers.extend(res["variants"])     # 同義詞/縮寫

            # 去重
            q["gold_answers"] = list(set(expanded_answers))

            fout.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"✅ 已生成 {'multi-hop' if only_multihop else '全部'} 擴充版 gold answers: {output_path}")
    print(f"📝 擴充 log 已寫入: {log_path}")


if __name__ == "__main__":
    expand_gold_answers()

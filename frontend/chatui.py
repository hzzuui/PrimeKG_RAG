# chatui.py
import streamlit as st
import requests

st.set_page_config(page_title="PrimeKG 知識問答")
st.title("PrimeKG 智慧問答助手")

display_mode = st.radio(
    "請選擇要顯示的回應內容：",
    ["RAG 回應", "不使用 RAG 回應", "同時顯示兩者"],
    index=0
)


# 對話歷史紀錄（一次存 query + rag/no_rag 回應）
if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("請輸入問題：", placeholder="例如：『幹細胞有哪些臨床應用？』")

if query:
    with st.spinner("生成回答中..."):
        try:
            res = requests.post("http://localhost:5000/rag", json={
                "query": query,
                "mode": display_mode  # "RAG 回應" or "不使用 RAG 回應" or "同時顯示兩者"
            })

            if res.status_code == 200:
                data = res.json()
                response_rag = data.get("answer_rag", "（RAG 無回應）")
                response_no_rag = data.get("answer_no_rag", "（No-RAG 無回應）")
            else:
                response_rag = response_no_rag = f"錯誤：{res.text}"

        except Exception as e:
            response_rag = response_no_rag = f"錯誤：{str(e)}"

        # ✅ 確保變數已定義後再加入紀錄
        st.session_state.history.append({
            "query": query,
            "response_rag": response_rag,
            "response_no_rag": response_no_rag
        })

# 顯示紀錄
st.markdown("### 對話紀錄")
for i, entry in enumerate(st.session_state.history, 1):
    st.markdown(f"**問題 {i}：** {entry['query']}")

    if display_mode == "RAG 回應":
        st.markdown(f"- **使用 RAG 回答：**\n\n{entry['response_rag']}")
    elif display_mode == "不使用 RAG 回應":
        st.markdown(f"- **未使用 RAG 回答：**\n\n{entry['response_no_rag']}")
    else:
        st.markdown(f"- **使用 RAG 回答：**\n\n{entry['response_rag']}")
        st.markdown(f"- **未使用 RAG 回答：**\n\n{entry['response_no_rag']}")
    st.markdown("---")

# 將屬性邏輯建成 NetworkX 有向圖（PAG）
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import pickle  

INPUT_PATH = "data/attr_implications.csv"
OUTPUT_GRAPH_GML = "data/attribute_graph.gml"     # 給 Gephi 或可視化工具用
OUTPUT_GRAPH_PKL = "data/attribute_graph.pkl"     # 給 Python 模組載入用

def build_attribute_graph():
    # 檢查檔案是否存在
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 找不到輸入檔案：{INPUT_PATH}")
        return

    # 讀取 A → B 關聯規則
    df = pd.read_csv(INPUT_PATH)

    # 建立 NetworkX 有向圖
    G = nx.DiGraph()

    # 加入節點與邊（含信心度）
    for _, row in df.iterrows():
        antecedent = row["antecedent"]
        consequent = row["consequent"]
        confidence = row["confidence"]

        G.add_edge(antecedent, consequent, weight=confidence)

    # 儲存為 GML 格式（可視覺化用）
    nx.write_gml(G, OUTPUT_GRAPH_GML)
    print(f"屬性關聯圖（GML）已儲存至：{OUTPUT_GRAPH_GML}")

    # 儲存為 Pickle 格式（Python 模組用）
    with open(OUTPUT_GRAPH_PKL, "wb") as f:
        pickle.dump(G, f)
    print(f"屬性關聯圖（Pickle）已儲存至：{OUTPUT_GRAPH_PKL}")

    # 顯示圖（可選）
    draw_graph(G)

def draw_graph(G):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color='lightblue',
        node_size=3000,
        font_size=10,
        arrows=True
    )
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()}
    )
    plt.title("🔗 Attribute Reasoning Graph (PAG)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    build_attribute_graph()

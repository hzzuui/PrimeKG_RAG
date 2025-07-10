import os
import pickle
import json
import networkx as nx

def convert_graph_pkl_to_json():
    # 定義資料路徑
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    pkl_path = os.path.join(data_dir, "attribute_graph.pkl")
    json_path = os.path.join(data_dir, "attribute_graph.json")

    # 讀取 .pkl 檔
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"找不到檔案：{pkl_path}")

    with open(pkl_path, "rb") as f:
        G = pickle.load(f)

    # 轉換成 dict 結構
    attr_graph = {}
    for source, target in G.edges():
        attr_graph.setdefault(source, []).append(target)

    # 儲存為 JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(attr_graph, f, indent=2, ensure_ascii=False)

    print(f"已成功轉換：{pkl_path}")
    print(f"輸出檔案：{json_path}")

if __name__ == "__main__":
    convert_graph_pkl_to_json()

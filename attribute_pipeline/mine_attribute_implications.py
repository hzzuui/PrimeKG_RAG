# 使用 Apriori/Association Rule 找出 A → B 關聯

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

INPUT_PATH = "data/entity_attributes.csv"
OUTPUT_PATH = "data/attr_implications.csv"

def mine_implications():
    # 讀取 entity-attribute binary 資料
    df = pd.read_csv(INPUT_PATH)
    
    # 移除 entity_id 欄位，只保留 0/1 欄位做分析
    df_attrs = df.drop(columns=["entity_id"])
    df_attrs = df_attrs.astype(bool)  # 必須轉成 bool，apriori 才能處理

    # 套用 Apriori 算法找頻繁屬性組合
    frequent_itemsets = apriori(df_attrs, min_support=0.01, use_colnames=True)

    # 從頻繁項組導出 association rules（邏輯推論）
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)

    # 過濾掉非單一對單一（只保留 A → B，不要 A,B → C 這種）
    rules = rules[
        (rules["antecedents"].apply(lambda x: len(x) == 1)) &
        (rules["consequents"].apply(lambda x: len(x) == 1))
    ]

    # 轉成 dataframe 格式
    results = pd.DataFrame({
        "antecedent": rules["antecedents"].apply(lambda x: list(x)[0]),
        "consequent": rules["consequents"].apply(lambda x: list(x)[0]),
        "confidence": rules["confidence"]
    })

    # 儲存結果
    results.to_csv(OUTPUT_PATH, index=False)
    print(f"屬性推論關係已儲存至：{OUTPUT_PATH}")

if __name__ == "__main__":
    mine_implications()

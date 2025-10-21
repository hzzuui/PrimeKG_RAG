# rule_generator.py
import pandas as pd
import json
from collections import defaultdict
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def mine_rules_from_json(triple_path, relation_filter, entity_keyword=None, 
                         min_support=0.01, min_conf=0.8, output_path=None):
    """根據指定的關係與關鍵詞動態生成規則庫"""
    with open(triple_path, "r", encoding="utf-8") as f:
        triples = json.load(f)

    # 篩選目標關係
    subset = [t for t in triples if t["relation_type"] == relation_filter]
    
    # 若指定關鍵詞（例如疾病名稱），可進一步篩選
    if entity_keyword:
        subset = [t for t in subset if entity_keyword.lower() in t["target_name"].lower() 
                  or entity_keyword.lower() in t["source_name"].lower()]
    if not subset:
        print("⚠️ 沒有找到符合條件的三元組")
        return pd.DataFrame()

    # 建立 transaction
    relation_dict = defaultdict(list)
    for item in subset:
        src = item["source_name"].strip()
        tgt = item["target_name"].strip()
        relation_dict[tgt].append(src)
    
    transactions = list(relation_dict.values())
    if len(transactions) < 5:
        print("⚠️ 資料太少，略過 Apriori")
        return pd.DataFrame()

    # 套用 Apriori
    encoder = TransactionEncoder()
    df_hot = pd.DataFrame(encoder.fit(transactions).transform(transactions), columns=encoder.columns_)
    frequent_itemsets = apriori(df_hot, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)

    rule_base = rules[["antecedents", "consequents", "support", "confidence"]]
    rule_base["antecedents"] = rule_base["antecedents"].apply(list)
    rule_base["consequents"] = rule_base["consequents"].apply(list)

    if output_path:
        rule_base.to_csv(output_path, index=False)
        print(f"💾 動態規則已輸出至 {output_path}")
    return rule_base

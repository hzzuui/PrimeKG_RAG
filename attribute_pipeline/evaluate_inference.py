#評估在 Step 5 中進行的屬性推論結果是否準確，與原始 Ground Truth（即 entity_attributes.csv）進行比較
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import os

# 設定輸入檔案位置
GROUND_TRUTH_PATH = "data/entity_attributes.csv"
INFERRED_PATH = "data/entity_attributes_inferred.csv"


def evaluate_inference():
    # 檢查檔案是否存在
    if not os.path.exists(GROUND_TRUTH_PATH) or not os.path.exists(INFERRED_PATH):
        print("找不到輸入檔案，請確認是否已完成前置步驟。")
        return

    # 讀取資料
    df_true = pd.read_csv(GROUND_TRUTH_PATH, index_col="entity_id")
    df_pred = pd.read_csv(INFERRED_PATH, index_col="entity_id")

    # 對齊兩者欄位與順序
    df_true = df_true[df_pred.columns]

    # 結果報表
    print("\n推論評估報告 (全體平均):")
    precision = precision_score(df_true, df_pred, average="micro")
    recall = recall_score(df_true, df_pred, average="micro")
    f1 = f1_score(df_true, df_pred, average="micro")

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\n各屬性分析:")
    for col in df_true.columns:
        p = precision_score(df_true[col], df_pred[col])
        r = recall_score(df_true[col], df_pred[col])
        f = f1_score(df_true[col], df_pred[col])
        print(f"{col:20} | Precision: {p:.4f} | Recall: {r:.4f} | F1: {f:.4f}")


if __name__ == "__main__":
    evaluate_inference()

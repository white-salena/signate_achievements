import pandas as pd

from collections import defaultdict, Counter
from itertools import combinations


# データ読み込み（例としてDataFrame作成）
df = pd.read_csv('competition/opt_2/dataset/train_A.tsv', sep='\t')

# 関連度スコアの計算
def compute_relevance(row):
    if row["event_type"] == 3 and row["ad"] == 1:  # 広告経由の購入
        return 3
    elif row["event_type"] == 2:  # 広告クリック
        return 2
    elif row["event_type"] == 1:  # 詳細ページ閲覧
        return 1
    else:  # カート追加
        return 0
    
df['weight'] = df.apply(compute_relevance, axis=1)

# 同日内の閲覧履歴に限定
df["time_stamp"] = pd.to_datetime(df["time_stamp"])
df["date"] = df["time_stamp"].dt.date

# Co-visitation マトリクス生成
pair_counter = defaultdict(int)

# ユーザー×日単位で処理
for (user_id, date), group in df.groupby(["user_id", "date"]):
    products = group[["product_id", "weight"]].drop_duplicates()

    # 全ての組み合わせを列挙（順序不要なので combinations）
    for (prod1, w1), (prod2, w2) in combinations(products.values, 2):
        key = tuple(sorted([prod1, prod2]))
        pair_counter[key] += w1 + w2

# 結果をDataFrameに変換
result = pd.DataFrame([
    {"product_id_1": k[0], "product_id_2": k[1], "co_visit_score": v}
    for k, v in pair_counter.items()
])

# スコア順にソート（任意）
result = result.sort_values("co_visit_score", ascending=False).reset_index(drop=True)

# 表示（上位10件）
print(result.head(10))

result.to_csv('competition/opt_2/dataset/result_save.tsv', sep='\t', index=False)

# 商品ごとのレコメンド候補（類似商品）をリストアップ
similar_items = defaultdict(list)

for _, row in result.iterrows():
    prod1 = row["product_id_1"]
    prod2 = row["product_id_2"]
    score = row["co_visit_score"]

    similar_items[prod1].append((prod2, score))
    similar_items[prod2].append((prod1, score))  # 対称性を考慮

# スコアでソート & 上位22件だけ取り出す
top_k = 22
for prod in similar_items:
    similar_items[prod] = sorted(similar_items[prod], key=lambda x: -x[1])[:top_k]

def recommend_for_user(viewed_products, similar_items, top_n=22):
    recommendation_scores = Counter()
    for product in viewed_products:
        for similar_product, score in similar_items.get(product, []):
            if similar_product not in viewed_products:
                recommendation_scores[similar_product] += score
    return recommendation_scores.most_common(top_n)
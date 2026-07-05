import pandas as pd

from collections import defaultdict, Counter
from itertools import combinations
from tqdm import tqdm


# =====================
# 1. データ読み込み
# =====================
train = pd.read_csv("competition/opt_2/dataset/train_A.tsv", sep="\t")
test = pd.read_csv("competition/opt_2/dataset/test.tsv", sep="\t")

# =====================
# 2. 前処理
# =====================
# 不要なデータ削除（例: event_type==3 かつ ad==0 を削除）
#train = train[~((train["event_type"] == 3) & (train["ad"] == 0))]
test_A = test[test['user_id'].str.endswith('_A')]

# =====================
# 3. Co-visitation Matrix 構築
# =====================
co_visit = defaultdict(Counter)

# 同一ユーザー & 同一日内のデータだけ使用（閲覧のまとまり）
train["time_stamp"] = pd.to_datetime(train["time_stamp"])
train["date"] = train["time_stamp"].dt.date

for (user, date), group in tqdm(train.groupby(["user_id", "date"])):
    items = group["product_id"].tolist()
    for a, b in combinations(set(items), 2):
        co_visit[a][b] += 1
        co_visit[b][a] += 1

# =====================
# 4. 類似商品リストに変換
# =====================
similar_items = {}
for item, neighbors in co_visit.items():
    sorted_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)
    similar_items[item] = sorted_neighbors[:50]  # トップ50件まで残す

# =====================
# 5. レコメンド関数定義
# =====================
def recommend_for_user(viewed_products, similar_items, top_n=22):
    score = Counter()
    for product in viewed_products:
        for sim_product, sim_score in similar_items.get(product, []):
            if sim_product not in viewed_products:
                score[sim_product] += sim_score
    return score.most_common(top_n)

# =====================
# 6. ユーザーの直近履歴から推薦
# =====================
user_recent = (
    train
    .sort_values("time_stamp")
    .groupby("user_id")["product_id"]
    .apply(lambda x: x[-5:].tolist())  # 直近5件
)

recommendations = {}

for user in tqdm(test_A["user_id"]):
    recent = user_recent.get(user, [])
    recs = recommend_for_user(recent, similar_items, top_n=22)
    recommendations[user] = [item for item, _ in recs]

# =====================
# 7. 結果を出力（提出用形式）
# =====================
output = []
for user_id, items in recommendations.items():
    for rank, product_id in enumerate(items, 1):
        output.append((user_id, product_id, rank))

output_df = pd.DataFrame(output, columns=["user_id", "product_id", "rank"])
output_df.to_csv("competition/opt_2/dataset/submission_A.tsv", sep='\t', index=False)

import pandas as pd

from surprise import SVD, Dataset, Reader

train = pd.read_csv('competition/opt/dataset/train_A.tsv', sep='\t')

test = pd.read_csv('competition/opt/dataset/test.tsv', sep='\t')
test_A = test[test['user_id'].str.endswith('_A')]

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

# 関連度スコアの設定
train['relevance'] = train.apply(compute_relevance, axis=1)

# 基準日（学習データの最終日）
end_date = pd.to_datetime("2017-04-30")

# time_stampをdatetime型に変換し、日付を抽出
train['date'] = pd.to_datetime(train['time_stamp']).dt.date
train['date'] = pd.to_datetime(train['date'])

# 減衰率を計算
train[f'weight_r_0.9'] = train['date'].apply(lambda x: 0.9 ** (end_date - x).days)

# 関連度 * 減衰率でスコア作成
train['score'] = train['relevance'] * train['weight_r_0.9']

# 評価(スコア)範囲の設定
reader = Reader(rating_scale=(0, 3))

# データの準備
data = Dataset.load_from_df(train[['user_id', 'product_id', 'score']], reader)

# 商品一覧の取得
unique_products = train['product_id'].drop_duplicates()

# 学習
trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)

# 予測
prediction_list = []

for user in test_A['user_id']:
    for product in unique_products:
        prediction = algo.predict(user, product)
        prediction_list.append({
            'user_id': user,
            'product_id': product,
            'score': prediction.est
        })

predictions = pd.DataFrame(prediction_list)

# 各ユーザーごとにスコア上位22件を抽出
top_22_per_user = (
    predictions
    .sort_values(['user_id', 'score'], ascending=[True, False])
    .groupby('user_id')
    .head(22)
    .reset_index(drop=True)
)

# 関連度ランクの計算
top_22_per_user['rank'] = (
    top_22_per_user
    .groupby('user_id')
    .cumcount() + 1
)

pd.DataFrame(top_22_per_user, columns=['user_id', 'product_id', 'rank']).to_csv('../dataset/predict2_A.tsv', sep='\t', index=False)
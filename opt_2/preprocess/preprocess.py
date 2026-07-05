import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

from module.Feature import UserFeature, ItemFeature


train = pd.read_csv('opt_2/data/train/train_A.tsv', sep='\t')

"""
train_cleaned = train[
    (train["event_type"] != 0) &
    ~((train["event_type"] == 3) & (train["ad"] == 0))
]
"""

train_user = UserFeature(train)
train_product = ItemFeature(train)

train_user.event_count()
train_product.event_count()

time_df = pd.DataFrame({
    'start_time': ['06:00', '10:00', '15:00', '18:00', '00:00'],
    'end_time': ['10:00', '15:00', '18:00', '23:59', '06:00'],
    'time_zone': ['morning', 'daytime', 'evening', 'night', 'latenight']
})

for _, row in time_df.iterrows():
    st = row['start_time']
    ed = row['end_time']
    tz = row['time_zone']

    train_user.time_count(st, ed, tz)
    train_product.time_count(st, ed, tz)

train_user.weekday_count()
train_product.weekday_count()

train_user.save('product_id', 'user_id', 'user')

train_product.save('user_id', 'product_id', 'product')

# relevance の取得のために user_df, product_df から必要な列だけに絞る
user_relevance = train_user.df[["user_id", "relevance"]].rename(columns={"relevance": "user_relevance"})
product_relevance = train_product.df[["product_id", "relevance"]].rename(columns={"relevance": "product_relevance"})

# まず user_relevance を join
interaction_df = train[["user_id", "product_id"]].drop_duplicates()
interaction_df = interaction_df.merge(user_relevance, on="user_id", how="left")

# 途中で一旦保存・確認
print(interaction_df.memory_usage(deep=True))
print(interaction_df.head())

# 次に product_relevance を join
interaction_df = interaction_df.merge(product_relevance, on="product_id", how="left")

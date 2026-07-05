import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from module.Tower import UserProductDataset, TwoTowerModel
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


# データ読み込み
user_df = pd.read_csv('opt_2/data/test_user_save.tsv', sep='\t')
product_df = pd.read_csv('opt_2/data/test_product_save.tsv', sep='\t')
#test = pd.read_csv('opt_2/data/test.tsv', sep='\t')
#test_A = test[test['user_id'].str.endswith('_A')]

# relevance は別にして、特徴量だけ残す
user_features = user_df.drop(columns=["user_id", "relevance"])
product_features = product_df.drop(columns=["product_id", "relevance"])

# IDを記録
user_ids = user_df["user_id"].values
product_ids = product_df["product_id"].values

# スケーリング（特徴量にNaNがない前提）
user_scaler = StandardScaler()
product_scaler = StandardScaler()
user_features_scaled = user_scaler.fit_transform(user_features)
product_features_scaled = product_scaler.fit_transform(product_features)

# IDと特徴量を紐づける
user_dict = dict(zip(user_ids, user_features_scaled))
product_dict = dict(zip(product_ids, product_features_scaled))

# relevance付きの組み合わせ（学習用）
# "relevance"が存在するすべての(user_id, product_id)ペアが必要
train_pairs = []
for _, row in user_df.iterrows():
    uid = row["user_id"]
    rel = row["relevance"]
    for _, prow in product_df.iterrows():
        pid = prow["product_id"]
        prel = prow["relevance"]
        # relevanceの組み合わせ例：ユーザー関連度 × 商品関連度の平均（仮）
        joint_relevance = (rel + prel) / 2
        train_pairs.append((uid, pid, joint_relevance))

train_dataset = UserProductDataset(train_pairs, user_dict, product_dict)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TwoTowerModel(user_features.shape[1], product_features.shape[1])
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(5):  # 必要に応じてエポック数変更
    model.train()
    total_loss = 0
    for user_x, product_x, y in train_loader:
        user_x, product_x, y = user_x.to(device), product_x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(user_x, product_x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")
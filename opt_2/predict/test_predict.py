import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import torch.nn as nn
import torch.optim as optim
from module.Tower import TwoTowerDataset, TwoTowerModel, recommend_topk_for_test_users
from torch.utils.data import DataLoader


# ---------- Load Data ----------
interaction_df = pd.read_csv("opt_2/data/interaction_A.tsv", sep="\t")  # user_id, product_id, relevance を含む
user_df = pd.read_csv("opt_2/data/test_user_save.tsv", sep="\t")
product_df = pd.read_csv("opt_2/data/test_product_save.tsv", sep="\t")
test = pd.read_csv('opt_2/data/test.tsv', sep='\t')
test_df = test[test['user_id'].str.endswith('_A')]
test_user_ids = test_df["user_id"].tolist()

# ---------- Training Setup (example only) ----------
dataset = TwoTowerDataset(interaction_df, user_df, product_df)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
model = TwoTowerModel(user_df.shape[1]-2, product_df.shape[1]-2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# for epoch in range(10):
model.train()
for user_x, product_x, y in loader:
    optimizer.zero_grad()
    pred = model(user_x, product_x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()

# ---------- Recommendation Output ----------
predictions = recommend_topk_for_test_users(model, test_user_ids, user_df, product_df, top_k=22, device="cpu")
pred_df = pd.DataFrame(predictions, columns=["user_id", "product_id", "score"])
pred_df.to_csv("predictions_A.tsv", sep="\t", index=False)
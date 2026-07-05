import torch
import torch.nn as nn

from torch.utils.data import Dataset


class TwoTowerDataset(Dataset):
    def __init__(self, interaction_df, user_df, product_df):
        self.user_df = user_df.set_index("user_id")
        self.product_df = product_df.set_index("product_id")
        self.interactions = interaction_df

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        user_id = row['user_id']
        product_id = row['product_id']
        label = row['relevance']

        user_features = self.user_df.loc[user_id].drop("relevance", errors="ignore").values.astype("float32")
        product_features = self.product_df.loc[product_id].drop("relevance", errors="ignore").values.astype("float32")

        return (
            torch.tensor(user_features),
            torch.tensor(product_features),
            torch.tensor(label, dtype=torch.float32)
        )

class Tower(nn.Module):
    def __init__(self, input_dim, embedding_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, x):
        return self.model(x)

class TwoTowerModel(nn.Module):
    def __init__(self, user_dim, product_dim, embedding_dim=64):
        super().__init__()
        self.user_tower = Tower(user_dim, embedding_dim)
        self.product_tower = Tower(product_dim, embedding_dim)

    def forward(self, user_x, product_x):
        user_emb = self.user_tower(user_x)
        product_emb = self.product_tower(product_x)
        return (user_emb * product_emb).sum(dim=1)  # 内積

def ndcg_at_k(r, k):
    r = r[:k]
    if not r.any():
        return 0.0
    dcg = (r / torch.log2(torch.arange(2, r.size(0) + 2, dtype=torch.float32))).sum()
    ideal_dcg = (torch.sort(r, descending=True).values / torch.log2(torch.arange(2, r.size(0) + 2, dtype=torch.float32))).sum()
    return dcg / ideal_dcg

def recommend_topk_for_test_users(model, test_user_ids, user_df, product_df, top_k=22, device="cpu"):
    model.eval()
    model.to(device)

    product_ids = product_df['product_id'].values
    product_features = torch.tensor(product_df.drop(columns=['product_id', 'relevance']).values, dtype=torch.float32).to(device)
    predictions = []

    with torch.no_grad():
        product_emb = model.product_tower(product_features)
        for user_id in test_user_ids:
            user_row = user_df[user_df['user_id'] == user_id]
            if user_row.empty:
                continue
            user_feature = torch.tensor(user_row.drop(columns=['user_id', 'relevance']).values, dtype=torch.float32).to(device)
            user_emb = model.user_tower(user_feature)
            scores = torch.matmul(user_emb, product_emb.T).squeeze(0)
            topk_indices = torch.topk(scores, top_k).indices.cpu().numpy()
            topk_scores = scores[topk_indices].cpu().numpy()

            for idx, score in zip(topk_indices, topk_scores):
                predictions.append((user_id, product_ids[idx], float(score)))

    return predictions
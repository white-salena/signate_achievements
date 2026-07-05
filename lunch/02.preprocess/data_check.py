import pandas as pd

# train = pd.read_csv('lunch/01.data/train_split.csv')
# test = pd.read_csv('lunch/01.data/test.csv')
# train_drop = train.drop(columns='y')
# df = pd.concat([train_drop, test], ignore_index=True)

df = pd.read_csv('lunch/01.data/df_kcal.csv')

# 基本情報
df.info()

# 統計情報
print(df.describe())

# 欠損行数
print(df.isnull().sum())

# kcal欠損のメニュー
print(df.loc[df['kcal'].isna(), 'name'])

# メニューの一覧
print(df["name"].unique())

# 特記事項の一覧
print(df["remarks"].unique())

# イベントの一覧
print(df["event"].unique())

# 天気の一覧
print(df["weather"].unique())
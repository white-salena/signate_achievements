import pandas as pd
import re

from lunch.module.lunch_module import name_split
from module.fileutil import read_data

train, test = read_data('lunch/01.data')

train_split = name_split(train)

prod_taken = train_split['y']
train_split_drop = train_split.drop(columns='y')

df = pd.concat([train_split_drop, test], ignore_index=True)

# kcal欠損の補完（同じcategoryの平均値）
# データリーク回避のため、一旦分離してtrainの平均値を取得してtestにも適用
category_train = df[df['datetime'] < '2014-10-01'].copy()
category_test = df[df['datetime'] >= '2014-10-01'].copy()

category_mean = category_train.groupby('category')['kcal'].mean()

category_train['kcal'] = category_train['kcal'].fillna(
    category_train['category'].map(category_mean)
)
category_test['kcal'] = category_test['kcal'].fillna(
    category_test['category'].map(category_mean)
)

kcal_df = pd.concat([category_train, category_test], ignore_index=True)

# 雨が降ってないときの降水量を0に変換
kcal_df['precipitation'] = (
    kcal_df['precipitation']
    .replace('--', 0)
    .astype(float)
)

# 給料日だけ1で他がNaNなので0に変換
kcal_df['payday'] = kcal_df['payday'].fillna(0).astype(int)

# weekとweatherとcategoryはOne-Hot Encoding
categorical_cols = [
    'week',
    'weather',
    'category'
]
kcal_df = pd.get_dummies(
    kcal_df,
    columns=categorical_cols,
    dtype=int
)

# remarksとeventは記載あり=1、なし=0で変換
kcal_df['remarks'] = kcal_df['remarks'].notna().astype(int)
kcal_df['event'] = kcal_df['event'].notna().astype(int)

# 年と月と日の特徴量を追加
kcal_df['year'] = kcal_df['datetime'].dt.year
kcal_df['month'] = kcal_df['datetime'].dt.month
kcal_df['day'] = kcal_df['datetime'].dt.day

# trainとtestに再分割
preprocessed_train = kcal_df[kcal_df['datetime'] <= '2014-09-30'].copy()
preprocessed_test = kcal_df[kcal_df['datetime'] >= '2014-10-01'].copy()

# 経過日数の特徴量を追加
preprocessed_train['days_from_start'] = (
    preprocessed_train['datetime'] - preprocessed_train['datetime'].min()
).dt.days
preprocessed_test['days_from_start'] = (
    preprocessed_test['datetime'] - preprocessed_test['datetime'].min()
).dt.days

# 不要なカラムを削除
preprocessed_train = preprocessed_train.drop(columns=['datetime', 'name'])
preprocessed_test = preprocessed_test.drop(columns=['datetime', 'name'])

# csv出力
preprocessed_train.to_csv('lunch/01.data/preprocessed_train.csv', index=False, encoding='utf-8-sig')
prod_taken.to_csv('lunch/01.data/prod_taken.csv', index=False, encoding='utf-8-sig')
preprocessed_test.to_csv('lunch/01.data/preprocessed_test.csv', index=False, encoding='utf-8-sig')
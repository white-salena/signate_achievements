import pandas as pd
import re

from module.fileutil import read_data

train, test = read_data('lunch/01.data')

new_rows = []
for _, row in train.iterrows():
    # 特記事項で判定
    matches = re.findall(r'(.+?)（(\d+)食）', str(row['remarks']))

    if len(matches) >= 2:
        for menu, num in matches:
            new_row = row.copy()
            new_row['name'] = menu.lstrip('、').strip()
            new_row['y'] = int(num)
            new_row['remarks'] = ''
            new_rows.append(new_row)
    else:
        new_rows.append(row)

train_split = pd.DataFrame(new_rows).reset_index(drop=True)

# 確認用にcsv出力(普段はコメントアウト)
# train_split.to_csv('lunch/01.data/train_split.csv', index=False, encoding='utf-8-sig')

prod_taken = train_split['y']
train_split_drop = train_split.drop(columns='y')

df = pd.concat([train_split_drop, test], ignore_index=True)

# datetimeを日付型に変換
df["datetime"] = pd.to_datetime(df["datetime"])

# 表記揺れ修正用
NAME_MAP = {
    # イカフライ
    '厚切イカフライ': '厚切りイカフライ',

    # ヒレカツ
    '手作りひれかつ': '手作りヒレカツ',
    'ヒレカツ': '手作りヒレカツ',

    # トンカツ
    '手作りトンカツ': '手作りロースカツ',

    # メンチ
    'クリームチーズ入りメンチ': 'クリームチーズ入りメンチカツ',
    'チーズメンチカツ': 'チーズ入りメンチカツ',

    # ハンバーグ
    'チーズハンバーグ': 'チーズ入りハンバーグ',
    '和風ハンバーグ': '和風ソースハンバーグ',

    # 唐揚げ
    '鶏の唐揚': '鶏の唐揚げ',
    '鶏肉の唐揚げ': '鶏の唐揚げ',

    # カシューナッツ
    '鶏のカッシュナッツ炒め': '鶏肉とカシューナッツ炒め',

    # 生姜焼き
    '豚肉の生姜焼': '豚肉の生姜焼き',
    'ポーク生姜焼き': '豚肉の生姜焼き',

    # 麻婆
    'マーボ豆腐': '麻婆豆腐',
    'マーボ茄子': '麻婆茄子',

    # 青椒肉絲
    '青椒肉絲': 'チンジャオロース',

    # 白身魚あんかけ
    '白身魚唐揚げ野菜あん': '白身魚唐揚げ野菜あんかけ',
    '白身魚唐揚げ野菜餡かけ': '白身魚唐揚げ野菜あんかけ',

    # カレイ
    'カレイ唐揚げ野菜餡かけ': 'カレイ唐揚げ野菜あんかけ',
    'カレイの唐揚げ': 'カレイ唐揚げ',

    # 牛すき
    '牛肉すき焼き風': '牛すき焼き風',
    '牛スキヤキ': '牛すき焼き風',
    '牛丼風': '牛丼風煮',

    # 豚しゃぶ
    '豚冷シャブ野菜添え': '豚の冷しゃぶ',
}

df['name'] = df['name'].replace(NAME_MAP)

def category(name):
    # ハンバーグ
    if 'ハンバーグ' in name:
        return 'ハンバーグ'

    # 揚げ物
    if any(x in name for x in [
        'カツ', 'フライ', '唐揚', 'から揚', 'コロッケ',
        '天ぷら', '南蛮', '包揚げ'
    ]):
        return '揚げ物'

    # 炒め物
    if any(x in name for x in [
        '炒め', '青椒', 'チンジャオ', '回鍋',
        'チャプチェ', 'プルコギ', '酢豚',
        '麻婆', 'マーボ', 'チャンプルー'
    ]):
        return '炒め物'

    # 煮物
    if any(x in name for x in [
        'シチュー', '煮', '肉じゃが',
        '筑前', '親子', 'ストロガノフ', 'ハヤシ'
    ]):
        return '煮物'

    # 焼き物
    if any(x in name for x in [
        '焼', 'ステーキ', 'ムニエル',
        'グリル', 'ソテー', 'フリカッセ'
    ]):
        return '焼き物'

    # 丼・ご飯
    if any(x in name for x in [
        '丼', '御飯', 'ご飯', 'ビュッフェ'
    ]):
        return 'ご飯'
    
    # カレー
    if 'カレー' in name:
        return 'カレー'

    return 'その他'

df['category'] = df['name'].apply(category)

# kcal欠損の補完（同じcategoryの平均値）
# データリーク回避のため、一旦分離してtrainの平均値を取得してtestにも適用
category_train = df[df["datetime"] < "2014-10-01"].copy()
category_test = df[df["datetime"] >= "2014-10-01"].copy()

category_mean = category_train.groupby("category")["kcal"].mean()

category_train["kcal"] = category_train["kcal"].fillna(
    category_train["category"].map(category_mean)
)
category_test["kcal"] = category_test["kcal"].fillna(
    category_test["category"].map(category_mean)
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
kcal_df["remarks"] = kcal_df["remarks"].notna().astype(int)
kcal_df["event"] = kcal_df["event"].notna().astype(int)

# 年と月と日の特徴量を追加
kcal_df["year"] = kcal_df["datetime"].dt.year
kcal_df["month"] = kcal_df["datetime"].dt.month
kcal_df["day"] = kcal_df["datetime"].dt.day

# trainとtestに再分割
preprocessed_train = kcal_df[kcal_df["datetime"] <= "2014-09-30"].copy()
preprocessed_test = kcal_df[kcal_df["datetime"] >= "2014-10-01"].copy()

# 経過日数の特徴量を追加
preprocessed_train["days_from_start"] = (
    preprocessed_train["datetime"] - preprocessed_train["datetime"].min()
).dt.days
preprocessed_test["days_from_start"] = (
    preprocessed_test["datetime"] - preprocessed_test["datetime"].min()
).dt.days

# 不要なカラムを削除
preprocessed_train = preprocessed_train.drop(columns=['datetime', 'name'])
preprocessed_test = preprocessed_test.drop(columns=['datetime', 'name'])

# csv出力
preprocessed_train.to_csv('lunch/01.data/preprocessed_train.csv', index=False, encoding='utf-8-sig')
prod_taken.to_csv('lunch/01.data/prod_taken.csv', index=False, encoding='utf-8-sig')
preprocessed_test.to_csv('lunch/01.data/preprocessed_test.csv', index=False, encoding='utf-8-sig')
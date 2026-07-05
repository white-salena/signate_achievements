import os
import glob
import pandas as pd

from sklearn.preprocessing import LabelEncoder


# 学習データの取得
def read_train_tsv(industry):
    file_path = f'../dataset/train_{industry}.tsv'

    return pd.read_csv(file_path, sep='\t')


# 予測データの取得
def read_test_tsv(industry):
    test = pd.read_csv('../dataset/test.tsv', sep='\t')

    extract_word = f'_{industry}'

    # 末尾が "_A" のテストデータだけを抽出して返す
    return test[test['user_id'].str.endswith(extract_word)]


# 学習データを訓練部分と検証部分に分割
def train_division(df: pd.DataFrame):
    # データのソート
    df = df.sort_values('time_stamp')

    # カラム名の取得
    features = [col for col in df.columns if col not in ['relevance', 'time_stamp', 'weekday', 'date']]

    # 学習用と検証用に日付で分割
    X = df[(df['time_stamp'] >= '2017-04-01') & (df['time_stamp'] <= '2017-04-23')]
    y = df[(df['time_stamp'] >= '2017-04-24') & (df['time_stamp'] <= '2017-04-30')]

    # 学習用データ作成
    X_train = X[features]
    y_train = X['relevance']

    # 検証用データ作成
    X_valid = y[features]
    y_valid = y['relevance']

    return X_train, y_train, X_valid, y_valid, features
    

# ユーザIDと商品IDのエンコード
class ColumnEncode():
    # ユーザIDのエンコード
    def user_encode(self, series: pd.Series):
        self.user_enc = LabelEncoder()

        return self.user_enc.fit_transform(series)
    
    # 商品IDのエンコード
    def product_encode(self, series: pd.Series):
        self.product_enc = LabelEncoder()

        return self.product_enc.fit_transform(series)
    
    # 予測データのエンコード
    def test_encode(self, series: pd.Series):
        return self.user_enc.transform(series)
    
    # ユーザIDのデコード
    def user_decode(self, series: pd.Series):
        return self.user_enc.inverse_transform(series)
    
    # 商品IDのデコード
    def product_decode(self, series: pd.Series):
        return self.product_enc.inverse_transform(series)

# 特徴量を追加した学習データの保管
def to_tsv_preprocess(df: pd.DataFrame, industry):
    file_path = f'../dataset/train_pre_{industry}.tsv'
    df.to_csv(file_path, sep='\t', index=False)

# 特徴量を追加した学習データの取得
def read_preprocess_tsv(industry):
    file_path = f'../dataset/train_pre_{industry}.tsv'

    return pd.read_csv(file_path, sep='\t')

# 個別の予測済みデータの保管
def to_one_predict(df: pd.DataFrame, industry):
    file_path = f'../dataset/predict_{industry}.tsv'
    df.to_csv(file_path, sep='\t', index=False)

# 予測データの保管
def to_all_predict(df: pd.DataFrame):
    # 保存フォルダのパス
    save_dir = '../dataset'

    # 既存ファイルのリストを取得
    existing_files = glob.glob(os.path.join(save_dir, "prediction_*.tsv"))

    # 次に保存するファイル名
    output_path = _get_next_prediction_filename(save_dir, existing_files)

    # 保存
    df.to_csv(output_path, sep="\t", index=False)

# 既存ファイルの最大番号を取得
def _get_next_prediction_filename(dir, files):
    numbers = []
    for f in files:
        # ファイル名から番号だけを抽出
        base = os.path.basename(f)
        name_part = base.replace("prediction_", "").replace(".tsv", "")
        if name_part.isdigit():
            numbers.append(int(name_part))
    next_num = max(numbers, default=0) + 1
    return os.path.join(dir, f"prediction_{next_num:03}.tsv")
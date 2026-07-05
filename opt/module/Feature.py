import pandas as pd


# ベースの特徴量生成クラス
class BaseFeature:
    def __init__(self, df: pd.DataFrame, col_name):
        self.df = df.copy()

        # タイムスタンプをdatetime型に変換
        self.df['time_stamp'] = pd.to_datetime(self.df['time_stamp'])

        # 曜日を追加
        self.df['weekday'] = self.df['time_stamp'].dt.day_name()

        # 減衰率カラムを追加
        self._attenuation_rate()

        # 関連度のカラム名
        self.rcol = col_name
    
    # 指定したカラム(ユーザIDか商品ID)毎に行動種別の集計(但し関連度を基準)
    def relevance_count(self, tarcol, type):
        # 指定された行動種別で絞る
        filtered_data = self.df[self.df[self.rcol] == type]

        # 作成する特徴量のカラム名を作成
        feature_name = f"{tarcol}_count_{type}"

        # 集計結果を返す
        return filtered_data.groupby([tarcol, self.rcol]).size().reset_index(name=feature_name)
    
    # 指定したカラム(ユーザIDか商品ID)毎に曜日単位で行動種別の集計(但し関連度を基準)
    def weekday_count(self, tarcol, type, wday):
        # 指定された行動種別でフィルタリング
        filtered_data = self.df[self.df[self.rcol] == type]

        # 指定された曜日でフィルタリング
        filtered_data = self.df[self.df['weekday'] == wday]

        # 作成する特徴量のカラム名を作成
        feature_name = f"{tarcol}_{wday}_count_{type}"
        
        # 集計結果を返す
        return filtered_data.groupby([tarcol, 'weekday']).size().reset_index(name=feature_name)
    
    # 減衰率の設定
    def _attenuation_rate(self):
        # 減衰率の設定
        r_values = [0.95, 0.9, 0.8]

        # 基準日（学習データの最終日）
        end_date = pd.to_datetime("2017-04-30")

        # time_stampをdatetime型に変換し、日付を抽出
        self.df['date'] = pd.to_datetime(self.df['time_stamp']).dt.date
        self.df['date'] = pd.to_datetime(self.df['date'])

        # 各減衰率ごとに重みを計算
        for r in r_values:
            self.df[f'weight_r_{r}'] = self.df['date'].apply(lambda x: r ** (end_date - x).days)


# ユーザ基準の特徴量生成クラス
class UserFeature(BaseFeature):
    def relevance_count(self):
        for i in range(4):
            count_data = super().relevance_count('user_id', i)

            # 結合と欠損値処理
            self.df = pd.merge(self.df, count_data, on=['user_id', self.rcol], how='left').fillna(0)
    
    def weekday_count(self):
        weekdays = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        for wday in weekdays:
            for i in range(4):
                count_data = super().weekday_count('user_id', i, wday)

                # 結合と欠損値処理
                self.df = pd.merge(self.df, count_data, on=['user_id', 'weekday'], how='left').fillna(0)


# 商品基準の特徴量生成クラス
class ItemFeature(BaseFeature):
    def relevance_count(self):
        for i in range(4):
            count_data = super().relevance_count('product_id', i)

            # 結合と欠損値処理
            self.df = pd.merge(self.df, count_data, on=['product_id', self.rcol], how='left').fillna(0)
    
    def weekday_count(self):
        weekdays = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        for wday in weekdays:
            for i in range(4):
                count_data = super().weekday_count('product_id', i, wday)

                # 結合と欠損値処理
                self.df = pd.merge(self.df, count_data, on=['product_id', 'weekday'], how='left').fillna(0)


# 関連度スコアの設定
def compute_relevance(row):
    if row["event_type"] == 3 and row["ad"] == 1:  # 広告経由の購入
        return 3
    elif row["event_type"] == 2:  # 広告クリック
        return 2
    elif row["event_type"] == 1:  # 詳細ページ閲覧
        return 1
    else:  # カート追加
        return 0

import pandas as pd


# ベースの特徴量生成クラス
class BaseFeature:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
    
        # タイムスタンプをdatetime型に変換
        self.df['time_stamp'] = pd.to_datetime(self.df['time_stamp'])

        # 曜日を追加
        self.df['weekday'] = self.df['time_stamp'].dt.day_name()

        # 関連度の特徴量生成
        self.df['relevance'] = self.df.apply(self.__compute_relevance, axis=1)

        # 減衰率の特徴量生成
        self.df['weight_r_0.9'] = self.__attenuation_rate()

        # スコア計算
        #self.df['score'] = self.df['relevance'] * self.df['weight_r_0.9']

    # 関連度スコアの計算
    def __compute_relevance(self, row):
        if row["event_type"] == 3 and row["ad"] == 1:  # 広告経由の購入
            return 3
        elif row["event_type"] == 2:  # 広告クリック
            return 2
        elif row["event_type"] == 1:  # 詳細ページ閲覧
            return 1
        else:  # カート追加
            return 0
    
    # 減衰率の計算
    def __attenuation_rate(self):
        # 基準日（学習データの最終日）
        end_date = pd.to_datetime("2017-04-30")

        # time_stampをdatetime型に変換し、日付を抽出
        self.df['date'] = pd.to_datetime(self.df['time_stamp']).dt.date
        self.df['date'] = pd.to_datetime(self.df['date'])

        # 減衰率を計算
        return self.df['date'].apply(lambda x: 0.9 ** (end_date - x).days)
    
    def save(self, drop_val, group_val, file_val):
        # いらないカラムを削除
        save_df = self.df.drop(columns=[drop_val, 'event_type', 'ad', 'time_stamp', 'weekday', 'date'])

        # 集約方法を定義
        agg_dict = {col: 'sum' for col in save_df.columns if col not in [group_val, 'weight_r_0.9']}
        agg_dict['weight_r_0.9'] = 'mean'

        # 集約
        grouped_df = save_df.groupby(group_val, as_index=False).agg(agg_dict)

        # ファイルパスを作成
        file_path = f'/home/whitesalena/python/pytorch/programs/Competition/project_signate/opt_2/data/test_{file_val}_save.tsv'

        # 保存
        grouped_df.to_csv(file_path, sep='\t', index=False)

class UserFeature(BaseFeature):
    def event_count(self):
        for i in range(4):
            # 行動種別で絞る
            filtered_data = self.df[self.df['event_type'] == i]

            # カラム名作成
            feature_name = f'user_event_count_{i}'

            # 集計
            event_scores = filtered_data.groupby(['user_id', 'event_type']).size().reset_index(name=feature_name)
            self.df = self.df.merge(event_scores, on=['user_id', 'event_type'], how='left').fillna(0)
    
    def time_count(self, start_time, end_time, time_zone):
        # 時間でフィルター
        temp_df = self.df[
            (self.df['time_stamp'].dt.time >= pd.to_datetime(start_time).time()) &
            ~(self.df['time_stamp'].dt.time < pd.to_datetime(end_time).time())
        ]

        for i in range(4):
            # 行動種別で絞る
            filtered_data = temp_df[temp_df['event_type'] == i]

            # カラム名作成
            feature_name = f'user_{time_zone}_count_{i}'

            # 集計
            event_scores = filtered_data.groupby(['user_id', 'event_type']).size().reset_index(name=feature_name)
            self.df = self.df.merge(event_scores, on=['user_id', 'event_type'], how='left').fillna(0)
    
    def weekday_count(self):
        weekdays = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

        for wday in weekdays:
            # 指定された曜日でフィルタリング
            weekday_df = self.df[self.df['weekday'] == wday]

            for i in range(4):
                # 行動種別で絞る
                filtered_data = weekday_df[weekday_df['event_type'] == i]

                # カラム名作成
                feature_name = f'user_{wday}_count_{i}'

                # 集計
                event_scores = filtered_data.groupby(['user_id', 'event_type']).size().reset_index(name=feature_name)
                self.df = self.df.merge(event_scores, on=['user_id', 'event_type'], how='left').fillna(0)


class ItemFeature(BaseFeature):
    def event_count(self):
        for i in range(4):
            # 行動種別で絞る
            filtered_data = self.df[self.df['event_type'] == i]

            # カラム名作成
            feature_name = f'product_event_count_{i}'

            # 集計
            event_scores = filtered_data.groupby(['product_id', 'event_type']).size().reset_index(name=feature_name)
            self.df = self.df.merge(event_scores, on=['product_id', 'event_type'], how='left').fillna(0)
    
    def time_count(self, start_time, end_time, time_zone):
        # 時間でフィルター
        temp_df = self.df[
            (self.df['time_stamp'].dt.time >= pd.to_datetime(start_time).time()) &
            ~(self.df['time_stamp'].dt.time < pd.to_datetime(end_time).time())
        ]

        for i in range(4):
            # 行動種別で絞る
            filtered_data = temp_df[temp_df['event_type'] == i]

            # カラム名作成
            feature_name = f'product_{time_zone}_count_{i}'

            # 集計
            event_scores = filtered_data.groupby(['product_id', 'event_type']).size().reset_index(name=feature_name)
            self.df = self.df.merge(event_scores, on=['product_id', 'event_type'], how='left').fillna(0)
    
    def weekday_count(self):
        weekdays = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        
        for wday in weekdays:
            # 指定された曜日でフィルタリング
            weekday_df = self.df[self.df['weekday'] == wday]

            for i in range(4):
                # 行動種別で絞る
                filtered_data = weekday_df[weekday_df['event_type'] == i]

                # カラム名作成
                feature_name = f'product_{wday}_count_{i}'

                # 集計
                event_scores = filtered_data.groupby(['product_id', 'event_type']).size().reset_index(name=feature_name)
                self.df = self.df.merge(event_scores, on=['product_id', 'event_type'], how='left').fillna(0)
import pandas as pd
import re


def name_split(df):
    new_rows = []

    for _, row in df.iterrows():
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
    
    return pd.DataFrame(new_rows).reset_index(drop=True)


class FeaturePreprocess():
    def __init__(self, df:pd.DataFrame):
        self.df = df
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])

    def name_replace(self):
        NAME_MAP = self._name_map()
        self.df['name'] = self.df['name'].replace(NAME_MAP)
    
    def _name_map(self):
        # 表記揺れ修正用の辞書
        return {
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
    
    def complement_kcal(self):
        self.df['category'] = self.df['name'].apply(self._category)

    def _category(self, name):
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

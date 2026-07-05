import os
import pandas as pd


def read_data(data_path):
    # train.csvとtest.csvのパス生成
    train_path = os.path.join(data_path, 'train.csv')
    test_path = os.path.join(data_path, 'test.csv')

    return pd.read_csv(train_path), pd.read_csv(test_path)


def read_predict_data(base_path):
    # 元ファイルまでのパス生成
    test_path = os.path.join(base_path, 'test.csv')

    # 前処理済みcsvファイルまでのパス生成
    preprpcessed_train_path = os.path.join(base_path, 'preprocessed_train.csv')
    preprpcessed_test_path = os.path.join(base_path, 'preprocessed_test.csv')
    prod_taken_path = os.path.join(base_path, 'prod_taken.csv')
    

    return pd.read_csv(preprpcessed_train_path), pd.read_csv(prod_taken_path), pd.read_csv(preprpcessed_test_path), pd.read_csv(test_path)

from module.fileutil import read_predict_data
from sklearn.ensemble import RandomForestRegressor


X_train, y_train, X_test, test = read_predict_data('lunch/01.data', 'y')

# ベースライン
base_model = RandomForestRegressor(n_estimators=200, random_state=42)

base_model.fit(X_train, y_train)

base_pred = base_model.predict(X_test)

# 提出用のデータ作成
base_submission = test[['datetime']].copy()
base_submission['y'] = base_pred
base_submission.to_csv('lunch/01.data/base_submission.csv', index=False, header=False, encoding='utf-8-sig')
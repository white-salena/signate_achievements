from module.fileutil import read_predict_data
from xgboost import XGBRegressor


X_train, y_train, X_test, test = read_predict_data('lunch/01.data')

xgb_model = XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

# 提出用のデータ作成
xgb_submission = test[['datetime']].copy()
xgb_submission['y'] = xgb_pred
xgb_submission.to_csv('lunch/01.data/xgb_submission1.csv', index=False, header=False, encoding='utf-8-sig')
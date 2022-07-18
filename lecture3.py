import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("diabetes.csv")

# 最初５つのデータを表示
#print(data[:5])

# x, yに分割
x = data.drop(["Outcome"], axis=1) # xにOutcome以外の値（特徴量）を代入
y = data["Outcome"]                # yにOutcomeの値を代入

# 学習用データとテスト用データに分離する
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True) # 2:8でiris.dataをテスト用、学習用データに分離する

# # SVMに学習データを投入する
# xgb = XGBClassifier()
# xgb.fit(x_train, y_train)

# # テスト、評価する
# y_pred = xgb.predict(x_test) # 学習機が予測した結果

# # 正解率
# print("正解率は？？", accuracy_score(y_test, y_pred))

# パラメータの検索
params = {"eta": [0.1, 0.3, 0.9], "max_depth": [2, 4, 6, 8]} # 汎化性能向上に利用するパラメータを定義

xgb_grid = GridSearchCV(
  estimator=XGBClassifier(),
  param_grid=params
)

xgb_grid.fit(x_train, y_train) # 学習

# paramsで定義したパラメータの中で最も汎化性能が高いパラメータをprintする
for key, value in xgb_grid.best_params_.items():
  print(key, value)

y_pred = xgb_grid.predict(x_test)

# 正解率
print("正解率は？？", accuracy_score(y_test, y_pred))
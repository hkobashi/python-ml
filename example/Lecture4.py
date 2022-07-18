import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier

data = pd.read_csv("titanic/train.csv")

print(data[:5])

# 必要のない列を削除する
data = data.drop(["Name", "Ticket", "Cabin"], axis=1)

# 欠損データの削除
data = data.dropna(subset=["Embarked"])

# ラベルを数値に変更しましょう
lb = LabelEncoder()
data.loc[:, "Sex"] = lb.fit_transform(data.loc[:, "Sex"].values)
data.loc[:, "Embarked"] = lb.fit_transform(data.loc[:, "Embarked"].values)

# x,yに分離する
x = data.drop(["Survived"], axis=1)
y = data["Survived"]

# 学習用データとテスト用データに分離する
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True)

# 学習する
xg = XGBClassifier()
xg.fit(x_train, y_train)

# テストする、評価する
# y_predには学習機が予測した結果が入る
y_pred = xg.predict(x_test)

# 答え合わせをして正解率を表示する
print("正解率は？？", accuracy_score(y_test, y_pred))

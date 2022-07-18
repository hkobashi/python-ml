import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

data = pd.read_csv("titanic/train.csv")
print(data[:5])

# 不要なカラムを削除する
data = data.drop(["Name", "Ticket", "Cabin"], axis=1)

# 欠損データの削除
data = data.dropna(subset=["Embarked"])

# ラベルを数値に変更
lb = LabelEncoder()
data.loc[:, "Sex"] = lb.fit_transform(data.loc[:, "Sex"].values)
data.loc[:, "Embarked"] = lb.fit_transform(data.loc[:, "Embarked"].values)

# x, yに分割
x = data.drop(["Survived"], axis=1) # xにSurvived以外の値（特徴量）を代入
y = data["Survived"]                # yにSurvivedの値を代入

# 学習用データとテスト用データに分離する
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True) # 2:8でdata.dataをテスト用、学習用データに分離する

# SVMに学習データを投入する
clf = SVC()
clf.fit(x_train, y_train) # 学習
#xg = XGBClassifier()
#xg.fit(x_train, y_train)


# テスト、評価する
#y_pred = xg.predict(x_test) # 学習機が予測した結果
y_pred = clf.predict(x_test) # 学習機が予測した結果

# 正解率
print("正解率は？？", accuracy_score(y_test, y_pred))
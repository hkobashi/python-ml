import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = pd.read_csv("iris.data", names=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Class"])

#print(iris)

# 入力データ、正解データに分離する
y = iris.loc[:, "Class"] # 
x = iris.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

#print(x)

# 学習用データとテスト用データに分離する
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True) # 2:8でiris.dataをテスト用、学習用データに分離する

# SVMに学習データを投入する
clf = SVC()
clf.fit(x_train, y_train) # 学習

# テスト、評価する
y_pred = clf.predict(x_test) # 学習機が予測した結果

# 正解率
print("正解率は？？", accuracy_score(y_test, y_pred))
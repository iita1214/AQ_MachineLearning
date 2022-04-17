import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


#csvファイルの読み込み
df = pd.read_csv("20220208CEJC_AS.csv")
# 説明変数の格納
x = df.loc[0:, ["ex_かな_percent"]]
# 目的変数の格納
y = df.loc[0:,"AQ_Ima"].astype("int64")
# トレーニングデータとテストデータに分割。
# test_size=0.3 : テストデータは30%、トレーニングデータ：70%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3) # random_state=None
# SVMを選択
model = svm.SVC()
# 学習
model.fit(x_train, y_train)
# トレーニングデータに対する精度
pred_train = model.predict(x_train)
accuracy_train = accuracy_score(y_train, pred_train)
error_train = mean_squared_error(y_train, pred_train)
print('トレーニングデータに対する正解率： %.2f' % accuracy_train)
print('トレーニングデータに対する平均二乗誤差： %.3f' % error_train)
# テストデータに対する精度
pred_test = model.predict(x_test)
accuracy_test = accuracy_score(y_test, pred_test)
error_test = mean_squared_error(y_test, pred_test)
print('テストデータに対する正解率： %.2f' % accuracy_test)
print('テストデータに対する平均二乗誤差： %.3f' % error_test)
print('トレーニングの推定値：', pred_train)
print(len(pred_train))
print('テストの推定値：', pred_test)
print(len(pred_test))

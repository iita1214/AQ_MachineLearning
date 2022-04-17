import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# csvファイルの読み込み
df = pd.read_csv("20220208CEJC_AS.csv")
# 説明変数の格納
x = df.loc[0:, ["ex_かな_percent","ex_や_percent",
                ]]
# ,"ex_よね_percent","ex_の_percent","ex_か_percent","ex_な_percent","ex_じゃん_percent","ex_かな_percent","ex_け_percent","ex_もん_percent","ex_もんね_percent","ex_わ_percent","ex_よな_percent","ex_かね_percent","ex_や_percent"
# 目的変数の格納
y = df.loc[0:,"AQ_Ima"].astype("int64")
# トレーニングデータとテストデータに分割。
# test_size=0.3 : テストデータは30%、トレーニングデータ：70%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=0)

model = Sequential()
# input_shape：説明変数の数　activation:活性化関数→シグモイド関数
model.add(Dense(units=10, input_shape=(len(x.keys()),), activation="sigmoid"))
model.add(Dense(units=10))
model.add(Dense(units=1))
# 損失関数は平均二乗誤差（mean_squared_error）、最適化アルゴリズムは確率的勾配降下法（sgd）
model.compile(loss="mean_squared_error", optimizer="sgd", metrics=['mae', 'mse'])
print(model.summary())
history = model.fit(x_train, y_train, epochs=20, batch_size=10, verbose=1, validation_split=0.1)
# トレーニングデータに対する推測
pred_train = model.predict(x_train)
# テストデータに対する推測
pred_test = model.predict(x_test)

print(pred_train)
print('訓練用データ数: '+str(len(pred_train)))
print(pred_test)
print('検証用データ数: '+str(len(pred_test)))
loss, train_mae, train_mse = model.evaluate(x_train, y_train, verbose=0)
print('訓練用データに対する平均絶対誤差： %.3f' % train_mae)
print('訓練用データに対する平均二乗誤差： %.3f' % train_mse)
loss, test_mae, test_mse = model.evaluate(x_test, y_test, verbose=0)
print('検証用データに対する平均絶対誤差： %.3f' % test_mae)
print('検証用データに対する平均二乗誤差： %.3f' % test_mse)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.show()

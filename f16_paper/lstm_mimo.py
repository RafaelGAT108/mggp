import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split


def create_sequences(database, lag):
    xs = np.array([database[i - lag:i, :] for i in range(lag, len(database))])
    ys = np.array([database[i, 2:5] for i in range(lag, len(database))])

    return xs, ys


df_train = pd.read_csv("../F16GVT_Files/BenchmarkData/F16Data_FullMSine_Level3.csv").to_numpy()
database_train = df_train[:, :5]

df_test = pd.read_csv("../F16GVT_Files/BenchmarkData/F16Data_FullMSine_Level2_Validation.csv").to_numpy()
database_test = df_test[:, :5]

lag_length = 5
X_train, y_train = create_sequences(database_train, lag_length)
X_test, y_test = create_sequences(database_test, lag_length)

print("Shape of X:", X_train.shape)
print("Shape of y:", y_train.shape)

model = Sequential([
    LSTM(32, activation='relu', input_shape=(lag_length, 5)),
    Dense(32, activation='relu'),
    Dense(3, activation='linear')
])

model.compile(optimizer='adam', loss='mse')

model.summary()

history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

loss = model.evaluate(X_test, y_test)
print(f"RMSE: {round(np.sqrt(loss), 6)}")

# predictions = model.predict(X_test)
# print("Predictions:", predictions)

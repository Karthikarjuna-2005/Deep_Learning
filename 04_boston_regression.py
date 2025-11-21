"""
04 - Neural network for predicting house prices using Boston Housing dataset
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import boston_housing
from sklearn.preprocessing import StandardScaler
import numpy as np

def build_model(input_shape):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main():
    (x_train,y_train),(x_test,y_test) = boston_housing.load_data()
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = build_model((x_train.shape[1],))
    model.summary()
    model.fit(x_train, y_train, validation_split=0.1, epochs=50, batch_size=16, verbose=0)
    loss, mae = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test MAE: {mae:.4f}")

if __name__ == '__main__':
    main()

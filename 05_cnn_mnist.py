"""
05 - Simple Convolutional Neural Network for MNIST handwritten digit classification
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Reshape(input_shape + (1,), input_shape=input_shape),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    (x_train, y_train),(x_test,y_test) = mnist.load_data()
    x_train = x_train.astype('float32')/255.0
    x_test = x_test.astype('float32')/255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    model = build_model(x_train.shape[1:], 10)
    model.summary()
    model.fit(x_train, y_train, validation_split=0.1, epochs=3, batch_size=128)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")

if __name__ == '__main__':
    main()

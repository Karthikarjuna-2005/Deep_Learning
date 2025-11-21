"""
03 - MLP for Reuters newswire classification (multi-class)
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def build_model(vocab_size, maxlen, num_classes):
    model = models.Sequential([
        layers.Embedding(vocab_size, 64, input_length=maxlen),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    vocab_size = 10000
    maxlen = 200
    (x_train,y_train),(x_test,y_test) = reuters.load_data(num_words=vocab_size)
    num_classes = max(y_train) + 1
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test = pad_sequences(x_test, maxlen=maxlen)
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    model = build_model(vocab_size, maxlen, num_classes)
    model.summary()
    model.fit(x_train, y_train, validation_split=0.1, epochs=5, batch_size=128)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")

if __name__ == '__main__':
    main()

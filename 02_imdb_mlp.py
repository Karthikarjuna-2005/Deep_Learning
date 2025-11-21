"""
02 - MLP for IMDB movie review binary classification (using built-in IMDB dataset)
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

def build_model(vocab_size, maxlen):
    model = models.Sequential([
        layers.Embedding(vocab_size, 32, input_length=maxlen),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    vocab_size = 10000
    maxlen = 200
    (x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=vocab_size)
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test = pad_sequences(x_test, maxlen=maxlen)
    model = build_model(vocab_size, maxlen)
    model.summary()
    model.fit(x_train, y_train, validation_split=0.1, epochs=3, batch_size=128)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")

if __name__ == '__main__':
    main()

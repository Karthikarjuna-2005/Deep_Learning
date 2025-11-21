"""
08 - One-hot encoding of words and characters (examples)
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

def words_one_hot(texts, num_words=1000):
    tok = Tokenizer(num_words=num_words, char_level=False)
    tok.fit_on_texts(texts)
    seq = tok.texts_to_sequences(texts)
    return tf.keras.utils.to_categorical(seq, num_classes=num_words)

def chars_one_hot(texts, maxlen=50):
    # character-level one hot using tf.one_hot on integer mapped chars
    chars = sorted(list({c for t in texts for c in t}))
    char_to_id = {c:i+1 for i,c in enumerate(chars)}  # reserve 0 for padding
    seqs = [[char_to_id[c] for c in t][:maxlen] for t in texts]
    seqs = tf.keras.preprocessing.sequence.pad_sequences(seqs, maxlen=maxlen)
    return tf.one_hot(seqs, depth=len(chars)+1)

def main():
    texts = ["hello world", "deep learning", "tensorflow examples"]
    wo = words_one_hot(texts, num_words=50)
    co = chars_one_hot(texts, maxlen=20)
    print("Words one-hot shape:", wo.shape)
    print("Chars one-hot shape:", co.shape)

if __name__ == '__main__':
    main()

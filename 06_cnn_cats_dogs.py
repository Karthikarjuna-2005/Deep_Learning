"""
06 - CNN for simple image classification (Dogs vs Cats).
Uses TensorFlow Datasets (cats_vs_dogs) if available, otherwise expects a directory with 'train' subfolders.
Training in this script is small-scale for demonstration.
"""
import tensorflow as tf
from tensorflow.keras import layers, models
import os

IMG_SIZE = 128
BATCH = 32
AUTOTUNE = tf.data.AUTOTUNE

def prepare_dataset():
    try:
        import tensorflow_datasets as tfds
        ds, info = tfds.load('cats_vs_dogs', split='train', with_info=True, as_supervised=True)
        ds = ds.map(lambda x,y: (tf.image.resize(x, (IMG_SIZE,IMG_SIZE))/255.0, tf.cast(y, tf.int32)))
        ds = ds.shuffle(1024).batch(BATCH).prefetch(AUTOTUNE)
        # split
        total = info.splits['train'].num_examples
        train = ds.take(total//BATCH - 50)  # small subset
        val = ds.skip(total//BATCH - 50).take(50)
        return train, val
    except Exception as e:
        print("tensorflow_datasets not available or failed. Please provide a directory of images structured for ImageDataGenerator.")
        dataset_dir = os.path.expanduser('~/cats_and_dogs/train')  # user-provided path
        train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.1)
        train = train_gen.flow_from_directory(dataset_dir, target_size=(IMG_SIZE,IMG_SIZE), batch_size=BATCH, subset='training')
        val = train_gen.flow_from_directory(dataset_dir, target_size=(IMG_SIZE,IMG_SIZE), batch_size=BATCH, subset='validation')
        return train, val

def build_model():
    model = models.Sequential([
        layers.Input((IMG_SIZE,IMG_SIZE,3)),
        layers.Conv2D(32,3,activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(64,3,activation='relu'),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    train, val = prepare_dataset()
    model = build_model()
    model.summary()
    model.fit(train, validation_data=val, epochs=3)
    print("Done.")

if __name__ == '__main__':
    main()

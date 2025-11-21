"""
07 - Use pre-trained VGG16 for transfer learning (image classification).
This script demonstrates how to use VGG16 as a feature extractor and attach a small head.
"""
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing import image_dataset_from_directory

IMG_SIZE = (224,224)
BATCH = 16

def build_model(num_classes):
    base = applications.VGG16(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    base.trainable = False
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    # For demonstration we expect a directory with subfolders per class
    dataset_dir = './data/transfer_train'  # user: replace with real path
    try:
        ds = image_dataset_from_directory(dataset_dir, image_size=IMG_SIZE, batch_size=BATCH, validation_split=0.2, subset='training', seed=123)
        val = image_dataset_from_directory(dataset_dir, image_size=IMG_SIZE, batch_size=BATCH, validation_split=0.2, subset='validation', seed=123)
    except Exception as e:
        print("Please provide a dataset directory in ./data/transfer_train with subfolders per class. Exiting.")
        return
    num_classes = ds.element_spec[1].shape[-1] if hasattr(ds.element_spec[1], 'shape') else ds.element_spec[1].shape[0]
    model = build_model(num_classes)
    model.summary()
    model.fit(ds, validation_data=val, epochs=3)
    print("Done.")

if __name__ == '__main__':
    main()

import tensorflow as tf
import matplotlib.pyplot as plt

data = tf.keras.preprocessing.image_dataset_from_directory("data")
data = data.map(lambda x, y: (x / 255.0, y))

# 2) Split into train/val/test
dataset_size = len(data)
train_size = int(dataset_size * 0.7)
val_size   = int(dataset_size * 0.2)
test_size  = dataset_size - train_size - val_size

train = data.take(train_size)
val   = data.skip(train_size).take(val_size)
test  = data.skip(train_size + val_size)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),     # 50% horizontal flips
    tf.keras.layers.RandomRotation(0.1),         # ±10% rotations
    tf.keras.layers.RandomZoom(0.1),             # ±10% zoom
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomTranslation(0.1,0.1),
    tf.keras.layers.RandomFlip("vertical"),
    tf.keras.layers.RandomTranslation(0.1,0.1),
    tf.keras.layers.RandomRotation(0.2),
])

train = (
    train
    .map(lambda x, y: (data_augmentation(x, training=True), y))
    .shuffle(1000)
    .cache()
)
val = val.cache()
test = test.cache()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), 1, activation="relu", input_shape=(256,256,3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(32, (3,3), 1, activation="relu"),
    tf.keras.layers.GlobalAveragePooling2D(),
    
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"]
)
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=1),
    tf.keras.callbacks.TensorBoard(log_dir="logs"),
]

history = model.fit(
    train,
    validation_data=val,
    epochs=20,
    callbacks=callbacks
)

plt.figure()
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.suptitle('Loss Curves')
plt.legend()
plt.show()



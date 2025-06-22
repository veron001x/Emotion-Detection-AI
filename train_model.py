from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

#  Disable GPU (force CPU only)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#  Define paths
train_dir = 'dataset/train'
val_dir = 'dataset/test'

#  Data preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical',
    color_mode='grayscale'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical',
    color_mode='grayscale'
)

#  Debug info to verify data
print("\nClass indices:", train_data.class_indices)
print("Training batches:", len(train_data))
print("Validation batches:", len(val_data))

#  Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotion classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#  Train with error handling
try:
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=20,
        verbose=1
    )

    # Save model
    if not os.path.exists("model"):
        os.makedirs("model")
    model.save("model/emotion_model.h5")
    print("\n Model training complete and saved as model/emotion_model.h5")

except Exception as e:
    print("\n Training crashed with error:")
    print(str(e))

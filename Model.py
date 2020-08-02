# =========================================
# FILE: Scratch.py
# DESC: CNN Built From Scratch
# AUTH: Matt B. Jackson
# =========================================

# Import Keras Framework
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# Image Dimensions
width, height = 96, 96

# Define Directories Where Data is Location
train_dir = "Images/train"
val_dir = "Images/val"
test_dir = "Images/test"

# Define Number of Epoch and Batch Size
epochs = 30
batch_size = 16
steps_per_epoch = 75

# Data Generator - Rescale Images
datagen = ImageDataGenerator(rescale=1./255)

# Training Data Generator
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode="categorical"
)

# Validation Data Generator
val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode="categorical"
)


# Create Model
model = Sequential()

model.add(Conv2D(filters=16, kernel_size=4, activation='relu', input_shape=(width, height, 3)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32, kernel_size=4, activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=4, activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=128, kernel_size=4, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(120, activation='softmax'))

# Compile Model
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=val_generator)

# Save for Future Predictions
model.save('Scratch.h5')

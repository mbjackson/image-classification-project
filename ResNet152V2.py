# =========================================
# FILE: ResNet152V2.py
# DESC: Create a Neural Network Using
#       ResNet152V2 as Pre-Trained Model
# AUTH: Matt B. Jackson
# =========================================

# Import Keras Framework
from keras.models import Model
from keras.applications import ResNet152V2
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense, BatchNormalization

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

# MobileNetV2 Model
base_model = ResNet152V2(include_top=False, input_shape=(width, height, 3), weights="imagenet")

# Additional Layers
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

# Output Layer
predictions = Dense(120, activation="softmax")(x)

# Construct Model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze Layers in Pre-Trained Model
for layer in base_model.layers:
    layer.trainable = False

# Compile Model
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=val_generator)

# Save for Future Predictions
model.save('ResNet152V2.h5')

# =========================================
# FILE: Predict.py
# DESC: Make Prediction from Given Model
# AUTH: Matt B. Jackson
# =========================================

# Import Keras Framework
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# Get User Input
file = input("Which Model Do You Want to Use: ")

# Load Model
model = load_model(file)

# Create Data Generator
datagen = ImageDataGenerator(rescale=1./255)

# Load Test Images
test_generator = datagen.flow_from_directory(
        'Images/test',
        target_size=(96, 96),
        batch_size=16,
        class_mode='categorical')

# Make Predictions and Gather Metrics
metrics = model.evaluate_generator(test_generator)

# Define Loss and Accuracy
loss = metrics[0]
accuracy = metrics[1] * 100

# Show Results to User
print("Loss: \t {}".format(loss))
print("Accuracy: \t {}%".format(round(accuracy, 2)))

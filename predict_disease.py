from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

def predict_disease(image_path):
    # Load and preprocess the image
    img = Image.open(image_path)
    img = img.resize((180, 180))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Return the predicted class
    return predicted_class

# Path to the new image
image_path = 'path_to_your_image.jpg'

# Load the model
model = load_model('C:\\Users\\Amir\\GardenV3\\my_model.h5')

# Predict the disease
predicted_class = predict_disease(image_path)

# Print the predicted class
print('Predicted class:', predicted_class)

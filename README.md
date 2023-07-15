# apple
AI and ML for Apple tree Monitoring

# Installation:
!pip install tensorflow

https://github.com/spMohanty/PlantVillage-Dataset

# create_dataset.py
This code will load the images, resize them to 180x180 pixels, and split them into a training set (80% of the images) and a validation set (20% of the images). The images and labels are batched together, so when you iterate over the dataset, you get a batch of 32 images and their corresponding labels.

# ResNet50.py
This code will load a pre-trained ResNet50 model, freeze the base model (so that the weights don't change during training), add a new classifier on top, compile the model (with an optimizer, a loss function, and metrics), and then train the model on our dataset.

# Validation.py
This will print out the accuracy of your model on the validation set, which is a good indicator of how well your model is likely to perform on unseen data.

# save_model
This will print out the accuracy of your model on the validation set, which is a good indicator of how well your model is likely to perform on unseen data.

# perdict_disease.py
This function takes the path to an image file, loads the image, resizes it to 180x180 pixels, normalizes the pixel values, adds an extra dimension to the array (because the model expects a batch of images, not a single image), and then feeds the image into the model to get a prediction. The function then returns the predicted class, which corresponds to the type of disease (or healthy, if the model predicts that the plant is healthy).

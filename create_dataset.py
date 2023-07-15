import tensorflow as tf

# Set the path to the directory containing the images
data_dir = '/path/to/your/data'

# Create a dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(180, 180),
  batch_size=32)

# The dataset will yield batches of images and labels. You can iterate over the dataset to see a batch.
for images, labels in dataset.take(1):
  print('images shape: ', images.shape)
  print('labels: ', labels)

# Assuming that validation_dataset is your validation data
validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(180, 180),
  batch_size=32)

# Evaluate the model on the validation data
loss, accuracy = model.evaluate(validation_dataset)

print("Validation accuracy: ", accuracy)
print("Validation loss: ", loss)

import tensorflow as tf

# Set the number of classes
num_classes = 5

# Load a pre-trained model
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',  # Load weights pre-trained on ImageNet
    input_shape=(180, 180, 3),
    include_top=False)  # Do not include the ImageNet classifier at the top

# Freeze the base model
base_model.trainable = False

# Add a new classifier on top
inputs = tf.keras.Input(shape=(180, 180, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(num_classes)(x)
model = tf.keras.Model(inputs, outputs)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# Train the model
model.fit(dataset, epochs=10)

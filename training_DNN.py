import tensorflow as tf
from keras import layers, Model

num_class = 4
input_shape = (224, 224, 3)

# Define input layer
inputs = layers.Input(shape=input_shape)

#Convolutional Layer
x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = layers.MaxPooling2D((2, 2))(x)
x1 = layers.Conv2D(64, (3, 3), activation='relu')(x)
x1 = layers.MaxPooling2D((2,2))(x1)
x2 = layers.Conv2D(128, (3, 3), activation='relu')(x1)
x2 = layers.MaxPooling2D((2,2))(x2)
x3 = layers.Conv2D(256, (3, 3), activation='relu')(x2)
x3 = layers.MaxPooling2D((2,2))(x3)
x4 = layers.Conv2D(512, (3, 3), activation='relu')(x3)

# Convert the 2D array feature to 1D vector for dense layer
# Method 1 - Flatten
# Mehod 2 - GlobalAveragePooling reduces number of parameters in Dense layer
x4 = layers.GlobalAveragePooling2D()(x4)
# x4 = layers.Flatten()(x4)
# x5 = layers.Dense(512, activation='relu')(x4)
output = layers.Dense(4, activation='softmax')(x4)

#Create model 
model = Model(inputs=inputs, outputs=output)

#compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )
model.summary()

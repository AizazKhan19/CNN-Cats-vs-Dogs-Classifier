import tensorflow as tf

# creating a model with 3 convolution layer

def create_and_compile_model():
    model = tf.keras.models.Sequential([

        # Input layer
        tf.keras.layers.Input(shape=(150, 150, 3)),

        # Rescaling layer to normalize the pixel values
        tf.keras.layers.Rescaling(1./255),

        # Adding 1st conv layer
        tf.keras.layers.Conv2D(16, (3, 3), activation= 'relu'),
        # Poolig Layer
        tf.keras.layers.MaxPooling2D(2, 2),

        # Adding 2nd conv layer
        tf.keras.layers.Conv2D(32, (3, 3), activation= 'relu'),
        # Poolig Layer
        tf.keras.layers.MaxPooling2D(2, 2),

        # Adding 3rd conv layer
        tf.keras.layers.Conv2D(64, (3, 3), activation= 'relu'),
        # Poolig Layer
        tf.keras.layers.MaxPooling2D(2, 2),

        # Flatten the data into 1D vector
        tf.keras.layers.Flatten(),

        # Adding 1 Hidden layers with 512 neurons
        tf.keras.layers.Dense(512, activation = 'relu'),

        # Adding Output layer with 1 neuron with activation function of sigmoid for binary classification
        tf.keras.layers.Dense(1, activation= 'sigmoid')
    ])

    model.compile(
        optimizer = 'adam',
        loss = 'binary_crossentropy',
        metrics = ['accuracy']
    )

    return model

if __name__ =="__main__":
    model = create_and_compile_model()
    print(model.summary())
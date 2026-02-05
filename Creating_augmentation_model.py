import tensorflow as tf

def augmentation_model():
    
    aug_model = tf.keras.Sequential([

        # adding augmentaion layers

        # input layer
        tf.keras.layers.Input(shape=(150, 150, 3)),

        #Random Flip layer
        tf.keras.layers.RandomFlip('horizontal'),

        # Random Roatation layer
        tf.keras.layers.RandomRotation(0.2, fill_mode = 'nearest'),

        # Random Translation layer
        tf.keras.layers.RandomTranslation(0.2, 0.2, fill_mode = 'nearest'),

        # Random Zoom layer
        tf.keras.layers.RandomZoom(0.2, fill_mode = 'nearest') 
    ])

    # return the model

    return aug_model


if __name__ == "__main__":
   augmented_model = augmentation_model()
   print("augmented model created")
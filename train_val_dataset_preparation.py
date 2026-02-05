import tensorflow as tf

# creating dataset for training and validation
# Below function will return two datasets. One is training dataset and the second will be validation dataset.

def train_val_dataset_preparation():

    training_dataset = tf.keras.utils.image_dataset_from_directory(
        directory = './data/Petimages',  
        image_size = (120, 120),
        batch_size = (128),
        label_mode = 'binary',
        validation_split = 0.15,
        subset = 'training',
        seed = 42

    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        directory = './data/PetImages',
        image_size = (120, 120),
        batch_size = (128),
        label_mode = 'binary',
        validation_split = 0.15,
        subset = 'validation',
        seed = 42
    )


    return training_dataset, validation_dataset

# creating training and validation datasets

if __name__ == "__main__":
    training_dataset, validation_dataset = train_val_dataset_preparation()

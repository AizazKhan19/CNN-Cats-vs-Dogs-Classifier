import tensorflow as tf
from train_val_dataset_preparation import train_val_dataset_preparation
from Callbacks import Earlystoppingcallback
from creating_and_compiling_model import create_and_compile_model

# getting training and validation dataset
training_dataset, validation_dataset = train_val_dataset_preparation()

# get untrained model 
model= create_and_compile_model()

# now lets train the model using fit method
def training_model():
    history = model.fit(
        training_dataset,
        epochs = 35,
        validation_data= validation_dataset,
        callbacks= [Earlystoppingcallback()]
    )

    model.save('Augmented_model_cat_dog_classifier.keras')
    print('Model saved successfully')


if __name__ == '__main__':
    training_model()
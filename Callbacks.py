import tensorflow as tf
from creating_and_compiling_model import create_and_compile_model

model= create_and_compile_model()
# Creating class of callback that will inturrupt/ Stop the training when our desired goal is reached
class Earlystoppingcallback (tf.keras.callbacks.Callback):
    # defining the function
    def on_epoch_end(self, epochs, logs=None):
        # the goal to write logic that stops our training when our training accuracy reached >=30% and validation accuracy reached >=20% . this metrics are choosen low because i dont have strong GPU to for training

        if logs['accuracy'] >= 0.3 and logs['val_accuracy'] >= 0.2:
            self.model.stop_training = True
            print("Training stopped because our critera is met")


if __name__ == "__main__":
    
    print('file successfully executed')

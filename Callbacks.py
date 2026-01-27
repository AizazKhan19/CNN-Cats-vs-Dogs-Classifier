import tensorflow as tf
from creating_and_compiling_model import create_and_compile_model

model= create_and_compile_model()
# Creating class of callback that will inturrupt/ Stop the training when our desired goal is reached
def callbacks():

    class Earlystoppingcallback (tf.keras.callbacks.Callback):
        # defining the function
        def on_epoch_end(self, epochs, logs=None):
            # the goal to write logic that stops our training when our training accuracy reached >=95% and validation accuracy reached >=80%

            if logs['accuracy'] >= 0.95 and logs['val_accuracy'] >= 0.80:
                self.model.stop_training = True
                print("Training stopped because our critera is met")


if __name__ == "__main__":
    callbacks()
    print('file successfully executed')

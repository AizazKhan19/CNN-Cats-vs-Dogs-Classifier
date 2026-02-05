import tensorflow as tf
from creating_and_compiling_model import create_and_compile_model

model= create_and_compile_model()
# Creating class of callback that will inturrupt/ Stop the training when our desired goal is reached
class Earlystoppingcallback (tf.keras.callbacks.Callback):
    # defining the function
    def on_epoch_end(self, epochs, logs=None):
        # acc >= 90% and val_acc>=90%

        if logs['accuracy'] >= 0.90 and logs['val_accuracy'] >= 0.90:
            self.model.stop_training = True
            print("Training stopped because our critera is met")


if __name__ == "__main__":
    
    print('file successfully executed')

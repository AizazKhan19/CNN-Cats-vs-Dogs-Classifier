import tensorflow as tf
from train_val_dataset_preparation import train_val_dataset_preparation
import numpy as np

training_dataset, _ = train_val_dataset_preparation()

# getting first batch of images and labels
def getting_sample_batch():
    for images, labels in training_dataset.take(1):
        sample_batch_images = images
        sample_batch_labels = labels

    print(f'Maximum pixel value of images: {np.max(sample_batch_images)}\n')
    print(f'Shape of batch of images:{sample_batch_images.shape}')
    print(f'Shape of batch of labels:{sample_batch_labels.shape}')

if __name__ == "__main__":
    getting_sample_batch()
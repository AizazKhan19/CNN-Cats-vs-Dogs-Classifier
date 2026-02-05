import  tensorflow as tf
import matplotlib.pyplot as plt
from Creating_augmentation_model import augmentation_model
from train_val_dataset_preparation import train_val_dataset_preparation


augmented_model = augmentation_model()
training_dataset, _ = train_val_dataset_preparation()

# getting sample images form training dataset batch 1
for images, labels in training_dataset.take(1):
    smpl_images= images


# taking sample image
sample_image = tf.keras.utils.array_to_img(smpl_images[0])
images = [sample_image]

def checking_augmentation_on_sample_images():
    # applying random augmentation
    for _ in range(3):
        image_aug = augmented_model(tf.expand_dims(sample_image, axis=0))
        image_aug = tf.keras.utils.array_to_img( tf.squeeze(image_aug))
        images.append(image_aug)


    # making canvas to show images
    fig, axs = plt.subplots(1, 4, figsize = (14, 7) )
    for ax, image, title in zip(axs, images, ['Original image', '1st Augmented', '2nd Augmented', '3rd Augmented']):
        ax.imshow(image, cmap= 'gray')
        ax.set_title(title)
        ax.axis('off')

    plt.show()


if __name__ == "__main__":
    checking_augmentation_on_sample_images()

   




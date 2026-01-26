import os
from image_organizer import cat_folder, dog_folder
import matplotlib.pyplot as plt
import tensorflow as tf

def visualization():
    
    # checking how many images of cats and dogs are in their respective folders

    print(f'There are {len(os.listdir(cat_folder))} images of cats in the folder {cat_folder}') 
    print(f'There are {len(os.listdir(dog_folder))} images of dog in the folder {dog_folder}') 


    # Lets plot some images of cats and dogs

    # First get the files names for cats and dogs

    cat_filenames =  [os.path.join(cat_folder, filename) for filename in os.listdir(cat_folder)]
    dog_filename = [ os.path.join(dog_folder, filename) for filename in os.listdir(dog_folder)]

    fig, axes= plt.subplots(2,4, figsize= (16, 8))
    fig.suptitle('Images of cats and dogs', fontsize= 18)


    # plotting the first 4 images of cat class
    for i, cat_image in enumerate(cat_filenames[:4]):
        img = tf.keras.utils.load_img(cat_image)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Cat {i}')
    
    # plotting the first 4 images of dog class
    for i, dog_image in enumerate(dog_filename[:4]):
        img = tf.keras.utils.load_img(dog_image)
        axes[1, i].imshow(img)
        axes[1, i].set_title(f'Dog {i}')
    

    plt.show()

if __name__ == "__main__":
    visualization()
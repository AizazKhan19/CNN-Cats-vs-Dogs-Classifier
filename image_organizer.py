import os
import shutil

# Root directory where images are stored
pet_images_path= 'data/PetImages'

# creating subfolders for cats and dogs
cat_folder = os.path.join(pet_images_path, 'Cat')
dog_folder = os.path.join(pet_images_path, 'Dog')

# now making function to oraganize images into their desired folders
def organize_images():
    os.makedirs(cat_folder, exist_ok =  True)
    os.makedirs(dog_folder, exist_ok =  True)

    # Moving images to their respective folders
    for img in os.listdir(pet_images_path):
        if img.startswith('cat'):
            shutil.move(os.path.join(pet_images_path, img), os.path.join(cat_folder, img))
        elif img.startswith('dog'):
            shutil.move(os.path.join(pet_images_path, img), os.path.join(dog_folder, img))

    print('Images have been organized into cat and dog folders')

if __name__ == "__main__":
    organize_images()
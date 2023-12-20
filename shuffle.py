import os
import random
import shutil

def select_random_images(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all image files in the input folder
    image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]

    # Shuffle the list of image files randomly
    random.shuffle(image_files)

    # Select the first 500 images and copy them to the output folder
    for image_file in image_files[:500]:
        shutil.copy(image_file, output_folder)
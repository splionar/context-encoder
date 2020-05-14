from glob import glob
import numpy as np
import os
from skimage.io import imread
import tensorflow as tf


def read_img_to_np(img_path):
    img = imread(img_path)
    img = img.astype(float)
    img/= 255.0
    
    return img


def collect_images_from_path(path):

    images = [
        read_img_to_np(img_path)
        for img_path in glob(os.path.join(path, "*"))
    ]

    print(f"There are {len(images)} images in the dataset.")

    return images


def create_input_groudtruth_generator(images, crop_function):

    def input_groudtruth_generator():
        for image in images:
            yield crop_function(image)
    
    return input_groudtruth_generator


def create_dataset(
    data_path,
    crop_function,
    batch_size=None,
    input_shape=(128,128,3),
    groundtruth_shape=(64,64,3),
    shuffle=True
):
    # Open all images
    images = collect_images_from_path(data_path)

    # Create a generator that crop images
    input_groudtruth_generator = create_input_groudtruth_generator(
        images, crop_function
    )

    # Create a tensorflow dataset
    dataset = tf.data.Dataset.from_generator(
        input_groudtruth_generator,
        (tf.float32, tf.float32),
        output_shapes=(
            tf.TensorShape(input_shape), 
            tf.TensorShape(groundtruth_shape)
        )
    )

    # Shuffle and batch if needed
    if shuffle:
        num_images = len(images)
        dataset = dataset.shuffle(num_images)

    if batch_size:
        dataset = dataset.batch(batch_size)

    return dataset

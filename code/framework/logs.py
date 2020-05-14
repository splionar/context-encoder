from matplotlib import pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, LambdaCallback, LearningRateScheduler, CSVLogger

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalized data
    """
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))

def create_tensorboard_logger(log_dir="logs"):
    return TensorBoard(
        log_dir=log_dir, 
        histogram_freq=0,
        profile_batch=0,
    )


def create_csv_logger(log_dir="logs"):
    return CSVLogger(
        os.path.join(log_dir, "logs.csv"), 
        separator=',', append=False
    )


def create_inpainted_image_logger(
    dataset, myModel, reconstruction_func, log_dir="logs", name="Reconstruction"):

    file_writer = tf.summary.create_file_writer(
        os.path.join(log_dir, "image")
    )

    def image_logger(epoch, logs):
        for input_data, _ in dataset:
            predicted_region = myModel.predict(input_data, steps=1)
            *predicted_region, = predicted_region

            input_data = input_data.numpy()
            *input_data, = input_data

            reconstructed_image = [
                reconstruction_func(input_image, predicted_image)
                for input_image, predicted_image in zip(input_data, predicted_region)
            ]

            reconstructed_image = np.stack(reconstructed_image, axis=0)
         
            for i in range(3):
                plt.imsave(
                    os.path.join(
                        log_dir, "image", name + "_example_{}.png".format(i)
                    ),
                    normalize(reconstructed_image[i,::])
                )
            
            break

        with file_writer.as_default():
            tf.summary.image(
                name, 
                reconstructed_image, 
                step=epoch
            )

    return LambdaCallback(on_epoch_end=image_logger)


def create_logs(
    model, train_dataset, val_dataset, reconstruction_func, log_dir):

    tensorboard_callback = create_tensorboard_logger(log_dir=log_dir)
    csv_callback = create_csv_logger(log_dir=log_dir)

    train_image_callback = create_inpainted_image_logger(
        train_dataset, model, reconstruction_func, 
        log_dir=log_dir, name="Train_Result"
    )

    val_image_callback = create_inpainted_image_logger(
        val_dataset, model, reconstruction_func, 
        log_dir=log_dir, name="Val_Result"
    )

    return [
        tensorboard_callback,
        csv_callback,
        train_image_callback,
        val_image_callback
    ]
    

def apply_on_batch(func):
    def wrapper(x,y):
        batch_size = x.shape[0]
        return np.stack(
            [func(x[i,::].numpy(), y[i,::]) for i in range(batch_size)],
            axis=0
        )
    
    return wrapper

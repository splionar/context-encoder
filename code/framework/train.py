from matplotlib import pyplot as plt
import numpy as np 
import os 
import tensorflow as tf 

from framework.logs import create_logs, create_inpainted_image_logger

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalized data
    """
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))

def set_optimizer_for_model(model, loss_func, learning_rate=1e-3):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss_func)


def train_model(
    model, 
    loss_func,
    train_dataset, 
    val_dataset, 
    reconstruction_func, 
    out_dir,
    learning_rate=1e-3,
    num_epochs=100000
    ):

    # Create an optimizer for the model with the loss function
    set_optimizer_for_model(
        model, loss_func, learning_rate=learning_rate
    )

    # Create various logs for visualization during training
    logging_callbacks = create_logs(
        model, train_dataset, val_dataset,
        reconstruction_func, os.path.join(out_dir, "logs")
    )

    # Create a checkpoint saver, to save the model
    checkpoint_path = os.path.join(out_dir, "model/weights.ckpt")

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1,
        save_freq=2220
    )
    
    # Train the model
    model.fit(
        train_dataset, 
        validation_data=val_dataset,
        epochs=num_epochs,
        callbacks=logging_callbacks + [cp_callback]
    )


def test_model(
    model, 
    test_dataset, 
    loss_func,
    reconstruction_func, 
    out_dir,
    num_images_to_save=5
    ):

    mean_loss = 0.0
        
    test_folder = os.path.join(out_dir, "test")

    model.load_weights(os.path.join(out_dir, "model/weights.ckpt")).expect_partial()

    
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    for data_idx, (input_image, groundtruth) in enumerate(test_dataset):
        predicted_region = model.predict(input_image)

        loss_value = loss_func(
            predicted_region, groundtruth
        )

        mean_loss += loss_value

        if data_idx < num_images_to_save:
            reconstructed_image = reconstruction_func(
                input_image[0,::].numpy(), predicted_region[0,::]
            )

            plt.imsave(
                os.path.join(
                    test_folder, 
                    "result_{:02d}.png".format(data_idx)
                ),
                normalize(reconstructed_image)
            )

    num_data   = data_idx + 1
    mean_loss /= num_data

    with open(os.path.join(test_folder, "logs.txt"), "w") as fid:
        fid.write("Average loss on the test set: {}".format(mean_loss))




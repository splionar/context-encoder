import argparse
from matplotlib import pyplot as plt
import os 

from framework.dataset import create_dataset
from framework.train import train_model, test_model

from homework6 import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="../dataset/")
    parser.add_argument("--out_dir", type=str, default="output")
    parser.add_argument("--num_epochs", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--test", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    # Create the model
    reconstruction_model = create_reconstruction_model() # task 1.2

    if not args.test:
        # Create the dataset using the crop function
        train_dataset = create_dataset(
            os.path.join(args.data_path, "train"), 
            crop_image, # task 1.1, 2.1
            batch_size=args.batch_size,
        )

        val_dataset = create_dataset(
            os.path.join(args.data_path, "val"), 
            crop_image, # task 1.1, 2.1
            batch_size=args.batch_size, 
        )

        # Train your model
        train_model(
            reconstruction_model,
            reconstruction_loss, # task 1.3
            train_dataset, 
            val_dataset,
            reconstruct_input_image,  # task 1.4
            out_dir=args.out_dir,
            learning_rate=1e-3,
            num_epochs=args.num_epochs
        )

    else:
        # Create the dataset using the crop function
        test_dataset = create_dataset(
            os.path.join(args.data_path, "test"),
            crop_image,
            batch_size=1,
            shuffle=False
        )

        # Test your model
        test_model(
            reconstruction_model, 
            test_dataset, 
            reconstruction_loss,
            reconstruct_input_image, 
            args.out_dir,
            num_images_to_save=5
        )

    
if __name__ == "__main__":
    main()
else:
    print("Importing from main.")

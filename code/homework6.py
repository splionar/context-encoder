import numpy as np 
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Dropout, Activation
from tensorflow.keras.models import Sequential

# Feel free to add as many imports as you need
from random import randint


def crop_image(img, random=False):
    """
    Takes an image as input and returns a copy of it with a 
    missing region, as well as the corresponding missing region

    INPUT:
    - img: a numpy array of size (heigh, width, 3) 
    - random: if true, the crop is taken at a random position,
    if false, the crop is taken at the center
    /!\ IGNORE FOR TASK 1

    OUTPUT:
    - img_with_a_hole: a numpy array of size (heigh, width, 3)
    - missing_region: a numpy array of size (64, 64, 3)

    HINT:
    For task 6, change the default random value to True
    """    
    h, w, _ = np.shape(img)
    crop_size = 64
    offset = 7

    img_with_a_hole = img.copy()
    
    if random==False:        
        start_j = int(h/4)-1
        start_i = int(w/4)-1
        end_j = start_j + crop_size
        end_i = start_i + crop_size
        missing_region = img[start_j:end_j, start_i:end_i].copy()
    else:
        start_j = randint(0,h-1-crop_size)
        start_i = randint(0,w-1-crop_size)
        end_j = start_j + crop_size
        end_i = start_i + crop_size        
        missing_region = img[start_j:end_j, start_i:end_i].copy()
    
    img_with_a_hole[start_j+offset:end_j-offset, start_i+offset:end_i-offset] = 0
        
    return img_with_a_hole, missing_region


def create_reconstruction_model():
    """
    Create a keras sequential model that reproduces figure 9.a
    of the paper

    OUTPUT:
    - model: a keras sequential model
    """
    model = Sequential()

    # Encoder
    model.add(Conv2D(64, kernel_size=4, strides=(2, 2), padding = 'same', input_shape=(128,128,3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(64, kernel_size=4, strides=(2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, kernel_size=4, strides=(2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(256, kernel_size=4, strides=(2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(512, kernel_size=4, strides=(2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    # Bottleneck
    model.add(Conv2D(4000, kernel_size=4, strides=(1,1), padding = 'valid'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))

    # Decoder
    model.add(Conv2DTranspose(512, kernel_size=4, strides=(2, 2), padding = 'valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(256, kernel_size=4, strides=(2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(128, kernel_size=4, strides=(2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(64, kernel_size=4, strides=(2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(3, kernel_size=4, strides=(2, 2), padding = 'same'))

    return model


def reconstruction_loss(predicted_region, groundtruth):
    """
    Computes the loss between the predicted region and the 
    corresponding groundtruth.

    INPUT: 
    - predicted_region: a tensor of shape (batch_size, 64, 64, 3)
    - groundtruth: a tensor of shape (batch_size, 64, 64, 3)

    OUTPUT:
    - loss_value: a tensor scalar

    HINT:
    Functions that might be useful (but you can use any tensorflow function you find
    useful, not necessarily those):
    - tf.reduce_mean
    - tf.square
    - tf.reduce_sum
    """
    offset = 7

    # overlap
    overlapping_mask = np.ones((64,64,3))
    h, w, _ = np.shape(overlapping_mask)
    n_overlap = (offset * w * 2 + offset * (h - 2*offset) * 2)*3
    overlapping_mask[offset:h-offset, offset:w-offset] = 0
    
    predicted_overlap = predicted_region * overlapping_mask
    groundtruth_overlap = groundtruth * overlapping_mask
    error_overlap = (predicted_overlap - groundtruth_overlap)
    
    # non-overlap
    mask_in = np.zeros((64,64,3))
    mask_in[offset:h-offset, offset:w-offset] = 1
    
    n_in = (h - 2*offset) * (w - 2*offset) * 3
    predicted_in = predicted_region * mask_in
    groundtruth_in = groundtruth * mask_in
    error_in = predicted_in - groundtruth_in
    
    loss_overlap = 10*tf.reduce_mean(tf.square(error_overlap)) / n_overlap
    loss_in =  tf.reduce_mean(tf.square(error_in)) / n_in
    
    loss = (loss_overlap + loss_in)/10 * (h * w *3)
    
    return loss


def reconstruct_input_image(input_data, predicted_region):
    """
    Combines an input image (with a hole), and a (predicted) missing region
    to produce a full image.

    INPUT:
    - input_data: a numpy array of size (height, width, 3)
    - predicted_region: a numpy array of size (64, 64, 3)

    OUTPUT:
    - full_image: a numpy array of size (height, width, 3)
    """
    offset = 7

    h, w, _ = np.shape(predicted_region)

    mask = np.sum(input_data, axis=2) == 0
    mask_i = np.sum(mask, axis=0)
    mask_j = np.sum(mask, axis=1)
    
    #print(list(np.where(mask_i == 50)[0])[0])
    i = np.where(mask_i == 50)[0][0] - offset
    j = np.where(mask_j == 50)[0][0] - offset

    full_image = input_data.copy()
    full_image[j:j+h, i:i+w] = predicted_region

    return full_image

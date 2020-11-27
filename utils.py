import tensorflow as tf
from PIL import Image
import time
import functools
import numpy as np
import matplotlib.pyplot as plt

def tensor_to_image(tensor):
    '''Convert tensor to numpy array'''
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)

    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

def load_img(path_to_img):
    '''Load image from path'''
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]

    return img

def imshow(image, title=None):
    '''Show image'''
    if len(image > 3):
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)

    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()
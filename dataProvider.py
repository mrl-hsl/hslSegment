import tensorflow as tf
import glob
import os
import cv2
import config as cfg
import numpy as np

def load_and_preprocess_image(image_path, label_path):
    image = tf.io.read_file(image_path)
    try:
        image = tf.image.decode_jpeg(image, channels=3)
    except tf.errors.InvalidArgumentError:
        print("Could not decode image:", image_path)
        return None, None
    image = tf.image.convert_image_dtype(image, tf.float32)

    label = tf.io.read_file(label_path)
    label = tf.image.decode_jpeg(label, channels=3)
    label = tf.image.convert_image_dtype(label, tf.float32)

    return image, label

def one_hot_encode(label):
    palette = np.array([
        [0., 0., 0.],
        [0., 255., 0.],
        [255., 0., 0.]
    ], dtype=np.float32)

    one_hot_maps = []
    for color in palette:
        class_map = tf.reduce_all(tf.equal(label, color), axis=-1)
        one_hot_maps.append(class_map)
    
    return tf.stack(one_hot_maps, axis=-1)

def random_flip_left_right(image, label):
    image = tf.image.random_flip_left_right(image)
    label = tf.image.random_flip_left_right(label)
    return image, label

def random_crop(image, label, crop_percent=0.6):
    # Randomly crop image and label
    cond_crop_image = tf.cast(tf.random_uniform([], maxval=2, dtype=tf.int32, seed=2), tf.bool)
    cond_crop_label = tf.cast(tf.random_uniform([], maxval=2, dtype=tf.int32, seed=2), tf.bool)

    shape = tf.cast(tf.shape(image), tf.float32)
    h = tf.cast(shape[0] * crop_percent, tf.int32)
    w = tf.cast(shape[1] * crop_percent, tf.int32)

    image = tf.cond(cond_crop_image, lambda: tf.image.random_crop(image, [h, w, 3], seed=2), lambda: image)
    label = tf.cond(cond_crop_label, lambda: tf.image.random_crop(label, [h, w, 3], seed=2), lambda: label)

    return image, label

train_images = glob.glob('./dataSet/train/images/*.jpg')
train_labels = glob.glob('./dataSet/train/labels/*.jpg')
test_images = glob.glob('./dataSet/test/images/*.jpg')
test_labels = glob.glob('./dataSet/test/labels/*.jpg')

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=len(train_images))
train_dataset = train_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.map(random_flip_left_right, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.map(one_hot_encode)
train_dataset = train_dataset.map(random_crop)
train_dataset = train_dataset.batch(8)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = test_dataset.map(load_and_preprocess_image)
test_dataset = test_dataset.map(one_hot_encode)
test_dataset = test_dataset.batch(8)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

train_iterator = train_dataset.make_initializable_iterator()
next_train_batch = train_iterator.get_next()

test_iterator = test_dataset.make_initializable_iterator()
next_test_batch = test_iterator.get_next()

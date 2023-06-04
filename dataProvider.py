import tensorflow as tf
import glob
import os
import cv2
import config as cfg
import numpy as np


crop_percent = 0.6
channels = [3, 3]

palette = cfg.color_palette


def _parse_image_function(example_proto):
    """
    Create a dictionary describing the features.
    """
    image_feature_description = {
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'depth': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.string),
        'image': tf.FixedLenFeature([], tf.string)
    }
    exm = tf.parse_single_example(example_proto, image_feature_description)
    image = tf.image.decode_jpeg(exm['image'])
    label = tf.image.decode_jpeg(exm['label'])
    # height = tf.cast(exm['height'], tf.int32)
    # width = tf.cast(exm['width'], tf.int32)
    image = tf.cast(image, dtype=tf.float32)
    label = tf.cast(label, dtype=tf.float32)
    return image, label


def _color(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:

    # x = tf.image.random_hue(x, 0.5)
    x = tf.image.random_saturation(x, 0.8, 1.2)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.7)
    x = tf.clip_by_value(x, 0, 255)
    # x = tf.image.per_image_standardization(x)
    return x, y


def _one_hot_encode(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """
    Converts mask to a one-hot encoding specified by the semantic map.
    """
    one_hot_map = []
    for class_name in palette:
        class_map = tf.cast(tf.reduce_all(
            tf.equal(y, palette[class_name]), axis=-1), tf.float32)
        one_hot_map.append(class_map)
    one_hot_map = tf.stack(one_hot_map, axis=-1)
    y = tf.cast(one_hot_map, tf.float32)
    return x, y


def _flip_left_right(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """
    Randomly flips image and mask left or right in accord.
    """
    x = tf.image.random_flip_left_right(x, seed=12)
    y = tf.image.random_flip_left_right(y, seed=12)

    return x, y


def _crop_random(image, mask):
    """
    Randomly crops image and mask in accord.
    """
    cond_crop_image = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32, seed=2), tf.bool)
    cond_crop_mask = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32, seed=2), tf.bool)

    shape = tf.cast(tf.shape(image), tf.float32)
    h = tf.cast(shape[0] * crop_percent, tf.int32)
    w = tf.cast(shape[1] * crop_percent, tf.int32)

    image = tf.cond(cond_crop_image, lambda: tf.random_crop(
        image, [h, w, channels[0]], seed=2), lambda: tf.identity(image))
    mask = tf.cond(cond_crop_mask, lambda: tf.random_crop(
        mask, [h, w, channels[1]], seed=2), lambda: tf.identity(mask))

    return image, mask


def _resize_data(image, mask):
    """
    Resizes images to specified size.
    """
    image = tf.expand_dims(image, axis=0)
    mask = tf.expand_dims(mask, axis=0)

    image = tf.image.resize_images(image, (240, 320))
    mask = tf.image.resize_nearest_neighbor(mask, (240, 320))

    image = tf.squeeze(image, axis=0)
    mask = tf.squeeze(mask, axis=0)

    return image, mask

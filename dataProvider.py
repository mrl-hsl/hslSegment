import tensorflow as tf
import glob
import os
import cv2
import config as cfg
import numpy as np

trainSet = tf.data.TFRecordDataset('./dataSet/train.tfrecords')
testSet = tf.data.TFRecordDataset('./dataSet/test.tfrecords')

def _parse_image_function(example_proto):
  image_feature_description = {
    'height': tf.FixedLenFeature([], tf.int64),
    'width': tf.FixedLenFeature([], tf.int64),
    'depth': tf.FixedLenFeature([], tf.int64),
    'label': tf.FixedLenFeature([], tf.string),
    'image': tf.FixedLenFeature([], tf.string)
  }
  exm = tf.parse_single_example(example_proto, image_feature_description)
  image = tf.image.decode_jpeg(exm["image"])
  label = tf.image.decode_jpeg(exm['label']) 
  image =tf.cast(image,dtype=tf.float32)
  label =tf.cast(label,dtype=tf.float32)
  return image, label 
palette = np.array([
        [  0.,   0.,   0.],
        [  0.,   255.,   0.],
        [  255.,   0.,   0.],
        ], dtype=np.uint8)

def _one_hot_encode(x,y):
    """
    Converts mask to a one-hot encoding specified by the semantic map.
    """
    one_hot_map = []
    for colour in palette:
        class_map =tf.cast( tf.reduce_all(tf.equal(y, colour), axis=-1),tf.float32)
        one_hot_map.append(class_map)
    one_hot_map = tf.stack(one_hot_map, axis=-1)
    y = tf.cast(one_hot_map, tf.float32)
    return x, y

def _flip_left_right(x: tf.Tensor, y: tf.Tensor)-> tf.Tensor:
        """
        Randomly flips image and mask left or right in accord.
        """
        x = tf.image.random_flip_left_right(x, seed=12)
        y = tf.image.random_flip_left_right(y, seed=12)

        return x, y

crop_percent=0.6
channels=[3, 3]
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

trainSet = trainSet.map(_parse_image_function)
trainSet = trainSet.map(_flip_left_right)
trainSet = trainSet.map(_one_hot_encode)

testSet = testSet.map(_parse_image_function)
# testSet = testSet.map(_flip_left_right)
testSet = testSet.map(_one_hot_encode)

trainSet = trainSet.batch(8)
testSet  = testSet.batch(8)

train_iterator = trainSet.make_initializable_iterator()
next_train_batch = train_iterator.get_next()

test_iterator = testSet.make_initializable_iterator()
next_test_batch = test_iterator.get_next()
import tensorflow as tf
import numpy as np
import glob
import os

if not os.path.exists(tfRAddress):
    os.makedirs(tfRAddress)

tf.enable_eager_execution()

tfRAddress = './'
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_example(image_bytes, label_bytes):
  feature = {
      'image': _bytes_feature(image_bytes),
      'label': _bytes_feature(label_bytes)
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))

writerTrain = tf.data.experimental.TFRecordWriter(tfRAddress+'train.tfrecords', 'wb')
writerTest = tf.data.experimental.TFRecordWriter(tfRAddress+'test.tfrecords', 'wb')

image_paths = glob.glob('img/*.jpg')
np.random.shuffle(image_paths)

for i, image_path in enumerate(image_paths):
  image = tf.io.read_file(image_path)
  label_path = image_path.split('.')[0] + '_labeld.jpg'
  label = tf.io.read_file(label_path)

  tf_example = image_example(image.numpy(), label.numpy())

  if i < len(image_paths) // 5:
    writerTest.write(tf_example.SerializeToString())
  else:
    writerTrain.write(tf_example.SerializeToString())

writerTrain.close()
writerTest.close()
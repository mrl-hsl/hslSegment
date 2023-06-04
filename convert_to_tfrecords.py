import tensorflow as tf
import cv2
import glob
tf.enable_eager_execution()

showSampel = False
tfRAddress = './'
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
  
def image_example(image_string, label_string):
  image_shape = tf.image.decode_jpeg(image_string).shape
  feature = {
      'height': _int64_feature(image_shape[0]),
      'width': _int64_feature(image_shape[1]),
      'depth': _int64_feature(image_shape[2]),
      'image': _bytes_feature(image_string),
      'label': _bytes_feature(label_string)
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))

writerTrain =tf.python_io.TFRecordWriter(tfRAddress+'train.tfrecords')
writerTest =tf.python_io.TFRecordWriter(tfRAddress+'test.tfrecords')
i = 0
for address in glob.glob('img/*.jpg'):
  i+=1
  if len(address.split('_'))==1:
    image = cv2.imread(address)
    label = cv2.imread(address.split('.')[0]+'_labeld.jpg')
    if showSampel==True:
      cv2.imshow("img",image)
      cv2.imshow("label",label)
      cv2.waitKey(0)
    image_string = cv2.imencode('.jpg', image)[1].tostring()
    label_string = cv2.imencode('.jpg', label)[1].tostring()
    tf_example = image_example(image_string, label_string)
    if i<200:
      writerTest.write(tf_example.SerializeToString())
    else:
      writerTrain.write(tf_example.SerializeToString())

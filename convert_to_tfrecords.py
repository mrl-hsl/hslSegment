import tensorflow as tf
import cv2
import glob
import os
import numpy as np

# Enable eager execution
tf.enable_eager_execution()

showSampel = False
tfRAddress = './'
writerTrain = tf.python_io.TFRecordWriter(tfRAddress + 'train.tfrecords')
writerTest = tf.python_io.TFRecordWriter(tfRAddress + 'test.tfrecords')
src_dir = './data/src_dir/'


# Count the number of images
data_counter = 0
for data_dir in os.listdir(src_dir):
    data_dir_path = os.path.join(src_dir, data_dir)
    for img in os.listdir(os.path.join(data_dir_path, 'image')):
        data_counter = data_counter + 1

cnt = 0  # This counter is used to split the data to 2 parts: train/ and test/
for data_dir in os.listdir(src_dir):
    data_dir_path = os.path.join(src_dir, data_dir)
    for img in os.listdir(os.path.join(data_dir_path, 'image')):
        cnt += 1
        img_cv = cv2.imread(os.path.join(data_dir_path, 'image', img))
        # img_cv = cv2.resize(img_cv, (320, 240),
        #                     interpolation=cv2.INTER_NEAREST)

        img = img.split('.')[0]
        label = cv2.imread(os.path.join(
            data_dir_path, 'label', img + '_labeled.png'))
        # label = cv2.resize(label, (320, 240), interpolation=cv2.INTER_NEAREST)

        if showSampel == True:
            cv2.imshow("img", img_cv)
            cv2.imshow("label", label)
            cv2.waitKey(0)
        image_string = cv2.imencode('.png', img_cv)[1].tostring()
        label_string = cv2.imencode('.png', label)[1].tostring()
        tf_example = image_example(image_string, label_string)
        if float(cnt)/float(data_counter) < 0.85:
            writerTrain.write(tf_example.SerializeToString())
        else:
            writerTest.write(tf_example.SerializeToString())

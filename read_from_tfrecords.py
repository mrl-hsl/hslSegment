import tensorflow as tf
import os
from data_provider import _parse_image_function
import cv2

SHOW_IMAGE = True
SAVE_IMAGE = False

TF_RECORD_FILES = [
                    'train.tfrecords',
                    'test.tfrecords',
                ]

# Create directories if they do not exist
if (SAVE_IMAGE):
    path = TF_RECORD_FILES[0].split('.')[0]
    path_exists = os.path.exists(path)
    if not path_exists:
        os.makedirs(path)

    image_path = os.path.join(path, 'image')
    image_path_exists = os.path.exists(image_path)
    if not image_path_exists:
        os.makedirs(image_path)

    label_path = os.path.join(path, 'label')
    label_path_exists = os.path.exists(label_path)
    if not label_path_exists:
        os.makedirs(label_path)

dataset = tf.data.TFRecordDataset(
    TF_RECORD_FILES
)
dataset = dataset.map(_parse_image_function)
train_set = dataset


train_iterator = train_set.make_initializable_iterator()
next_train_batch = train_iterator.get_next()


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def show_label(label, cnt):
    label = label.astype('uint8')
    label = cv2.cvtColor(label, cv2.COLOR_RGB2BGR)
    if (SHOW_IMAGE):
        cv2.imshow('label', label)
        cv2.waitKey(0)

    if (SAVE_IMAGE):
        cv2.imwrite(label_path + '/' + str(cnt) + '.png', label)


def show_image(image, cnt):
    image = image.astype('uint8')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if (SHOW_IMAGE):
        cv2.imshow('image', image)
        cv2.waitKey(0)

    if (SAVE_IMAGE):
        cv2.imwrite(image_path + '/' + str(cnt) + '.png', image)


sess.run(train_iterator.initializer)

cnt = 0
while True:
    cnt += 1
    b = sess.run(next_train_batch)
    image = b[0]
    show_image(image, cnt)
    label = b[1]
    show_label(label, cnt)
    print(cnt)

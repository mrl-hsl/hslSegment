import tensorflow as tf
import cv2
import numpy as np
import config as cfg
import glob

n_steps = 15000
model_path = './models/' + str(n_steps) + '/meta'
test_mode = 1 # 0: webcam, 1: read from a directory, 2: read from tfrecords
color_palette = cfg.color_palette
num_classes = len(color_palette)
srd_dir = './data/src_dir/data_set1/image'

# Set config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Load the network
saver = tf.train.import_meta_graph(model_path + '/saved_model.meta')
saver.restore(sess, tf.train.latest_checkpoint(model_path))

# Set graphs to call
graph = tf.get_default_graph()
input = graph.get_tensor_by_name("input:0")
prediction = graph.get_tensor_by_name("predictor/predictions_argmax:0")

def evaluate(img):
    retval	=cv2.TickMeter()
    retval.start()
    out = sess.run(prediction, feed_dict={input: [img]})
    retval.stop()
    it = retval.getTimeMilli()
    print('inference time: {:.2f} ms'.format(round(it, 2)))
    # print("output shape:", out.shape)
    out = out[0]
    # pro = pro[0]
    label = np.zeros((240, 320, 3), np.uint8())
    for i in range(0, 240):
        for j in range(0, 320):
            # print(i, j, pro[i][j])
            label_index = out[i][j]
            for class_name, class_index in zip(color_palette, range(num_classes)):
                if label_index == class_index:
                    label[i][j] = color_palette[class_name]
                    break
    return label

def main():
    img_counter = 0
    if (test_mode == 0):    
        # Go for live
        cam = cv2.VideoCapture(0)
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            img_counter += 1
            img_cv = cv2.resize(frame, (320, 240))
            # img = (img - img.mean()) / img.std()
            # print(img.shape)
            print('image number {:d}'.format(img_counter))
            label = evaluate(img_cv)
            cv2.imshow('frame', img_cv)
            cv2.imshow('label', label)
            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k % 256 == 32:
                # SPACE pressed
                img_name = "opencv_frame_{}.png".format(img_counter)
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
            print('###############################################')
        cam.release()
        cv2.destroyAllWindows()
    elif (test_mode == 1):
        for img_name in glob.glob(srd_dir + '/*'):
            img_cv = cv2.imread(img_name)
            img_cv = cv2.resize(img_cv, (320, 240))
            img_counter += 1
            print('image number {:d}'.format(img_counter))
            label = evaluate(img_cv)
            cv2.imshow('frame', img_cv)
            cv2.imshow('label', label)
            k = cv2.waitKey(0)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k % 256 == 32:
                # SPACE pressed
                img_name = "opencv_frame_{}.png".format(img_counter)
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
            print('###############################################')
    elif (test_mode == 2):
        pass

if __name__ == '__main__':
    main()

import cv2
import numpy as np
import os
import shutil
import config as cfg

src_dir = './data/src_dir/syntech/label'
dst_dir = './data/src_dir/syntech/label_2'

data_cnt = 0
for data_dir in os.listdir(src_dir):
    data_cnt += 1


dst_dir_path_exists = os.path.exists(dst_dir)
if dst_dir_path_exists:
    # os.rmdir(dst_dir)
    shutil.rmtree(dst_dir)

dst_dir_path_exists = os.path.exists(dst_dir)
if not dst_dir_path_exists:
    os.makedirs(dst_dir)

color_palette = cfg.color_palette

cnt = 0  # This counter is used to split the data to 2 parts: train/ and test/
for img_name in os.listdir(src_dir):
    img_cv = cv2.imread(os.path.join(src_dir, img_name))
    height, width, channels = img_cv.shape
    for i in range(height):
        for j in range(width):
            pixel = img_cv[i][j]
            if np.array_equal(pixel, np.array([0, 0, 255])):
                img_cv[i][j] = color_palette['line']
    cnt += 1
    # cv2.imshow('figure', img_cv)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join(dst_dir, img_name), img_cv)

    print("\033[A")
    print(str(int((cnt/data_cnt)*100)) + ' %', end='')

# -*- coding:utf-8 -*-
import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

cwd = './data/'
classes={'yibao', 'kele'}
writer = tf.python_io.TFRecordWriter("bottle_train.tfrecords")

for index, name in enumerate(classes):
    class_path = cwd+name+'/'
    for img_name in os.listdir(class_path):
        img_path = class_path+img_name #每一张图片的地址
        img = Image.open(img_path)
        img = img.resize((128, 128))
        img_raw = img.tobytes() #将图片转化为二进制格式
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            "img_raw": tf.train.Feature()
        }))
        #序列化为字符串
        writer.write(example.SerializeToString())
writer.close()
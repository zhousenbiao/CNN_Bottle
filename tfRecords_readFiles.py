# -*- coding:utf-8 -*-

#读取TFRECORD文件

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()

    # 返回文件名和文件
    _, serialized_example = reader.read(filename_queue)

    # 将image数据和label取出来
    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'img_raw': tf.FixedLenFeature([], tf.string),
    })

    img = tf.decode_raw(features['img_raw'], tf.unit8)
    # reshape为128*128的3通道图片
    img = tf.reshape(img, [128, 128, 3])
    # 在流中抛出img张量
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return img, label
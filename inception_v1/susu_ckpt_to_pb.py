# -*-coding: utf-8 -*-
"""
    @Project: tensorflow_models_nets
    @File   : convert_pb.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-08-29 17:46:50
    @info   :
    -通过传入 CKPT 模型的路径得到模型的图和变量数据
    -通过 import_meta_graph 导入模型中的图
    -通过 saver.restore 从模型中恢复图中各个变量的数据
    -通过 graph_util.convert_variables_to_constants 将模型持久化
"""
import os

import tensorflow as tf
from create_tf_record import *
from tensorflow.python.framework import graph_util

resize_height = 224  # 指定图片高度
resize_width = 224  # 指定图片宽度
depths = 3


def freeze_graph_test(pb_path, image_path):
    '''
    :param pb_path:pb文件的路径
    :param image_path:测试图片的路径
    :return:
    '''
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # 定义输入的张量名称,对应网络结构的输入张量
            # input:0作为输入图像,keep_prob:0作为dropout的参数,测试时值为1,is_training:0训练参数
            input_image_tensor = sess.graph.get_tensor_by_name("input:0")
            input_keep_prob_tensor = sess.graph.get_tensor_by_name("keep_prob:0")
            # print(input_keep_prob_tensor)
            input_is_training_tensor = sess.graph.get_tensor_by_name("is_training:0")

            # 定义输出的张量名称
            output_tensor_name = sess.graph.get_tensor_by_name("InceptionV1/Logits/SpatialSqueeze:0")

            # 读取测试图片
            im = read_image(image_path, resize_height, resize_width, normalization=True)
            im = im[np.newaxis, :]
            # 测试读出来的模型是否正确，注意这里传入的是输出和输入节点的tensor的名字，不是操作节点的名字
            # out=sess.run("InceptionV3/Logits/SpatialSqueeze:0", feed_dict={'input:0': im,'keep_prob:0':1.0,'is_training:0':False})
            out = sess.run(output_tensor_name, feed_dict={input_image_tensor: im,
                                                          input_keep_prob_tensor: 1.0,
                                                          input_is_training_tensor: False})
            print("out:{}".format(out))
            score = tf.nn.softmax(out, name='pre')
            class_id = tf.argmax(score, 1)
            print("pre class_id:{}".format(sess.run(class_id)))
            with open("D:\PaperCode\ACE-2\susu_tensorflow_V1_learning\dataset\label.txt", mode='r') as F:
                labels = F.readlines()
                print(f"pre class name:{labels[sess.run(class_id)[0]]}")



def freeze_graph(input_checkpoint, output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    # output_node_names = "InceptionV3/Logits/SpatialSqueeze"
    output_node_names = 'InceptionV1/Logits/SpatialSqueeze'
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点

        # for op in sess.graph.get_operations():
        #     print(op.name, op.values())


def freeze_graph2(input_checkpoint, output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "InceptionV1/Logits/SpatialSqueeze"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点

        # for op in graph.get_operations():
        #     print(op.name, op.values())


if __name__ == '__main__':

    input_checkpoint = 'D:\PaperCode\ACE-2\susu_tensorflow_V1_learning\models\\best_models_24600_0.9766.ckpt'      # 输入ckpt模型路径
    out_pb_path = "D:\PaperCode\ACE-2\susu_tensorflow_V1_learning\models\susu_V1_0.9766.pb"               # 输出pb模型的路径
    # freeze_graph(input_checkpoint, out_pb_path)        # 调用freeze_graph将ckpt转为pb

    # 测试pb模型
    imagefile = "D:\PaperCode\ACE-2\susu_tensorflow_V1_learning\dataset\\test"
    imagefiles_name = os.listdir(imagefile)
    for name in imagefiles_name:
        files_path = os.path.join(imagefile, name)
        images_name = os.listdir(files_path)
        for image_name in images_name:
            image_path = os.path.join(files_path, image_name)
            print(image_path)
            freeze_graph_test(pb_path=out_pb_path, image_path=image_path)
    # image_path = 'D:\PaperCode\susu_tensorflow_V1_learning\predict_test\wolf in snow\\23.png'
    # freeze_graph_test(pb_path=out_pb_path, image_path=image_path)















# #输出网络节点w和b
# import tensorflow as tf
# import os
#
# variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
# def txt_save(data, output_file):
#     file = open(output_file, 'a')
#     for i in data:
#         s = str(i) + '\n'
#         file.write(s)
#     file.close()
#
#
# def network_param(input_checkpoint, output_file=None):
#     saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
#     with tf.Session() as sess:
#         saver.restore(sess, input_checkpoint)
#         variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
#         for i in variable:
#             print(i)     # 打印
#         txt_save(variables, output_file)  # 保存txt   二选一
#
# if __name__ == '__main__':
#     checkpoint_path = 'D:\PaperCode\susu_tensorflow_V1_learning\models\\best_models_18600_0.9922.ckpt'
#     output_file = '/models/susu_V1_0.9922_w_b.txt'
#     if not os.path.exists(output_file):
#         network_param(checkpoint_path, output_file)
import PIL.Image
import tensorflow as tf
import os
import numpy as np
import re
from PIL import Image
# import matplotlib.pyplot as plt
import cv2


class NodeLookup(object):
    def __init__(self):
        label_lookup_path = 'D:\PaperCode\ACE-1\susu-petrained-model\susu_inceptionv1_label.pbtxt'
        uid_lookup_path = 'D:\PaperCode\ACE-1\susu-petrained-model\susu-inceptionv1-label.txt'
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        # 加载分类字符串n************对应分类名称的文件
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        # 一行一行读取数据
        for line in proto_as_ascii_lines:
            # 去掉换行符
            line = line.strip('\n')
            # 按照‘\t’分割
            parsed_items = line.split('\t')
            # 获取分类编号
            uid = parsed_items[0]
            # 获取分类名称
            human_string = parsed_items[1]
            # 保存编号字符串n********与分类名称的映射关系
            uid_to_human[uid] = human_string
        # 加载分类字符串n**********对应分类编号1-1000的文件
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        node_id_to_uid = {}
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                # 获取分类编号1-1000
                target_class = int(line.split(':')[1])
            if line.startswith('  target_class_string:'):
                # 获取编号字符串n*********
                target_class_string = line.split(':')[1]
                # 保存分类编号1-1000与编号字符串n*******映射关系
                node_id_to_uid[target_class] = target_class_string[2:-2]
        # 建立分类编号1-1000对应分类名称的映射关系
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            # 获取分类名称
            name = uid_to_human[val]
            node_id_to_name[key] = name
        return node_id_to_name

    # 传入分类编号1-1000返回分类名称
    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


# 创建一个图来放google训练好的模型
# with tf.gfile.FastGFile('D:/PaperCode/susu_pth_ACE-1/Pretrained_google_inceptionv1/tensorflow_inception_graph.pb', 'rb') as f:
# with tf.gfile.FastGFile('D:\PaperCode\ACE-1\susu-petrained-model\pth_susu-pratrained_inceptionv1.pb', 'rb') as f:
with tf.gfile.FastGFile('D:\PaperCode\PanJinquan_tensorflow_model_learning\models\susu_tensorflow_v1_0.9922_model.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
    # print(graph_def)

with tf.Session() as sess:
    # softmax_tensor = sess.graph.get_tensor_by_name('Concat_209:0')
    softmax_tensor = sess.graph.get_tensor_by_name('InceptionV1/Logits/SpatialSqueeze:0')
    # 遍历目录
    for root, dirs, files in os.walk('D:\PaperCode\inception_three classes\Inception\susu_datasets3\predict_test'):
        for file in files:
            # 载入图片
            # image_data = tf.gfile.FastGFile(os.path.join(root, file), 'rb').read()
            # image_data = PIL.Image.open(os.path.join(root, file))
            image_data = cv2.imread((os.path.join(root, file)))
            image_data = tf.image.resize_images(image_data, [224,224], method=0)
            # image_data = tf.transpose(image_data, perm=[2, 0, 1])  
            image_data = tf.expand_dims(image_data, axis=0)
            # image_data = np.expand_dims(image_data, axis=0)
            image_data = image_data.eval()
            # image_data = tf.image.decode_jpeg(image_data)
            print("image_data:", image_data.shape)


            # predictions = sess.run(softmax_tensor, {'input:0': image_data})  # 图片格式是jpg格式
            predictions = sess.run(softmax_tensor, feed_dict={'input:0': image_data, 'keep_prob:0': 1.0, 'is_training:0': False})  # 图片格式是jpg格式
            # predictions = np.array(predictions)
            # predictions = tf.nn.softmax(predictions)
            # predictions = sess.run(predictions)
            print("11111111111:",predictions.shape)
            if predictions.shape[0] != 0:
                predictions = predictions[0, :]
            predictions = np.squeeze(predictions)  # 把结果转换为1维数据
            # predictions = tf.nn.softmax(predictions)

            # predictions = np.array(predictions).flatten()
            # print(predictions.shape)

            # 打印图片路径及名称
            print()
            image_path = os.path.join(root, file)
            print(image_path)
            # 显示图片
            # img = Image.open(image_path)
            # plt.imshow(img)
            # plt.axis('off')
            # plt.show()

            # 排序
            top_k = predictions.argsort()[-3:][::-1]
            node_lookup = NodeLookup()
            for node_id in top_k:
                # 获取分类名称
                human_string = node_lookup.id_to_string(node_id)
                # 获取该分类的置信度
                score = predictions[node_id]
                # print("node_id:", node_id)
                print('%s (score=%.5f)' % (human_string, score))
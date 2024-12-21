"""
@Time ： 2023/7/19 14:18
@Auth ： Luminous
@Contact:  3199409618@qq.com
@File ：computing similarity.py
@IDE ：PyCharm
"""
import tensorflow as tf
import pickle
import os
import numpy as np

def euclidean_distance_by_tf(vector1, vector2):
    # print(tf.square([1, 2, 3, 4]))
    # return vector1 - vector2
    with tf.Session() as sess:
     return tf.sqrt(tf.reduce_sum(tf.square(vector1 - vector2))).eval(session=sess)

def Manhattan_Distance_by_tf(vector1, vector2):
    with tf.Session() as sess:
     return tf.reduce_sum(tf.abs(tf.add(vector1, tf.negative(vector2)))).eval(session=sess)


def tf_cosine_distance(tensor1, tensor2):
    """
    consine相似度：用两个向量的夹角判断两个向量的相似度，夹角越小，相似度越高，得到的consine相似度数值越大
    数值范围[-1,1],数值越大越相似。
    :param tensor1:
    :param tensor2:
    :return:
    """
    # 把张量拉成矢量，这是我自己的应用需求
    tensor1 = tf.reshape(tensor1, shape=(1, -1))
    tensor2 = tf.reshape(tensor2, shape=(1, -1))

    # 求模长
    tensor1_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor1)))
    tensor2_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor2)))

    # 内积
    tensor1_tensor2 = tf.reduce_sum(tf.multiply(tensor1, tensor2))
    cosin = tensor1_tensor2 / (tensor1_norm * tensor2_norm)
    with tf.Session() as sess:
     return cosin.eval(session=sess)

def main(root1, root2, root_output):
    pkl_concept_1 = os.listdir(root1)
    pkl_concept_2 = os.listdir(root2)
    for m in range(len(pkl_concept_1)):
        for n in range(len(pkl_concept_2)):
            cav_1 = os.path.join(root1, pkl_concept_1[m])
            cav_2 = os.path.join(root2, pkl_concept_2[n])
            with open(cav_1, 'rb') as f:
                cav1_value = pickle.load(f)
                cav1_value = tf.convert_to_tensor(cav1_value, dtype='float32')
            with open(cav_2, 'rb') as f:
                cav2_value = pickle.load(f)
                cav2_value = tf.convert_to_tensor(cav2_value, dtype='float32')
            # print(f'euclidean_distance:similarity_{m}_{n}={euclidean_distance_by_tf(cav1_value, cav2_value)}')
            # print(f'Manhattan_Distance_by_tf:similarity_{m}_{n}={Manhattan_Distance_by_tf(cav1_value, cav2_value)}')
            # print(f'tf_cosine_distance_{m}_{n}={tf_cosine_distance(cav1_value, cav2_value)}\n')
            if not os.path.exists(root_output):
                os.makedirs(root_output)
            with open(os.path.join(root_output, root1.split("\\")[-1]+"-"+root2.split("\\")[-1]+".txt"), "a") as f:
                f.write(f'euclidean_distance:similarity_{m}_{n}_{pkl_concept_1[m]}_{pkl_concept_2[n]}={euclidean_distance_by_tf(cav1_value, cav2_value)}\n'
                        f'Manhattan_Distance_by_tf:similarity_{m}_{n}_{pkl_concept_1[m]}_{pkl_concept_2[n]}={Manhattan_Distance_by_tf(cav1_value, cav2_value)}\n'
                        f'tf_cosine_distance_{m}_{n}_{pkl_concept_1[m]}_{pkl_concept_2[n]}={tf_cosine_distance(cav1_value, cav2_value)}\n\n')

    # for i in range(len(pkl_dics_1)):
    #     pkl_concept_1_path = os.path.join(root1, pkl_dics_1[i])
    #     pkl_concept_1 = os.listdir(pkl_concept_1_path)
    #     for j in range(len(pkl_dics_2)):
    #         pkl_concept_2_path = os.path.join(root2, pkl_dics_2[j])
    #         pkl_concept_2 = os.listdir(pkl_concept_2_path)
    #         for m in range(len(pkl_concept_1)):
    #             for n in range(len(pkl_concept_2)):
    #                 cav_1 = os.path.join(pkl_concept_1_path, pkl_concept_1[m])
    #                 cav_2 = os.path.join(pkl_concept_2_path, pkl_concept_2[n])
    #                 with open(cav_1, 'rb') as f:
    #                     cav1_value = pickle.load(f)
    #                     cav1_value = tf.convert_to_tensor(cav1_value, dtype='float32')
    #                     # print(type(cav1_value))
    #                 with open(cav_2, 'rb') as f:
    #                     cav2_value = pickle.load(f)
    #                     cav2_value = tf.convert_to_tensor(cav2_value, dtype='float32')
    #                     # print(cav2_value)
    #                 print(f'euclidean_distance:similarity_{i}_{j}_{m}_{n}={euclidean_distance_by_tf(cav1_value, cav2_value)}')
    #                 print(f'Manhattan_Distance_by_tf:similarity_{i}_{j}_{m}_{n}={Manhattan_Distance_by_tf(cav1_value, cav2_value)}')
    #                 print(f'tf_cosine_distance_{i}_{j}_{m}_{n}={tf_cosine_distance(cav1_value, cav2_value)}\n')
                    # if not os.path.exists(root_output):
                    #     os.makedirs(root_output)
                    # with open(os.path.join(root_output, root1.split("\\")[-1]+"-"+root2.split("\\")[-1]+".txt"), "a") as f:
                    #     f.write(f'euclidean_distance:similarity_{i}_{j}_{m}_{n}_{pkl_concept_1[m]}_{pkl_concept_2[n]}={euclidean_distance_by_tf(cav1_value, cav2_value)}\n'
                    #             f'Manhattan_Distance_by_tf:similarity_{i}_{j}_{m}_{n}_{pkl_concept_1[m]}_{pkl_concept_2[n]}={Manhattan_Distance_by_tf(cav1_value, cav2_value)}\n'
                    #             f'tf_cosine_distance_{i}_{j}_{m}_{n}_{pkl_concept_1[m]}_{pkl_concept_2[n]}={tf_cosine_distance(cav1_value, cav2_value)}\n\n')
    print("-" * 30)
    print("finish computing")


if __name__ == '__main__':
    root1 = 'D:\PaperCode\ACE-1\susu_new\CAV_end\wolf_in_snow_back'
    root2 = 'D:\PaperCode\ACE-1\susu_new\CAV_end\Eskimo_dog_back'
    root_output = 'D:\PaperCode\ACE-1\susu_new\CAV_similarity'
    main(root1, root2, root_output)



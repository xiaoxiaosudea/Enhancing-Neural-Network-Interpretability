import time
import tensorflow as tf
import numpy as np
import cv2
import os

if __name__ == '__main__':
    patches_root = "D:\PaperCode\ACE-2\susu_new\ImageNet\\verify\shark\\0001\\add"
    save_image_root = "D:\PaperCode\ACE-2\susu_new\ImageNet\\verify\shark\\0001\DQN-10"
    image_root = "D:\PaperCode\ACE-2\susu_new\ImageNet\\verify\shark\\0001\saved\\0001.png"
    image = cv2.imread(image_root)
    print(image.shape)
    image = cv2.resize(image, (2240, 2240))
    patches_files = os.listdir(patches_root)
    i = 0
    j = 0
    for patch_file in patches_files:
        concept_path = os.path.join(patches_root, patch_file)
        concept_path1 = os.listdir(concept_path)
        for patch in concept_path1:
            patch_path = os.path.join(concept_path, patch)
            patch = cv2.imread(patch_path)
            patch = cv2.resize(patch, (112, 112))

            d = 0
            if i * 112 > image.shape[1]:
                i = 0
                j += 1
            if j * 112 > image.shape[0]:
                i = 0
                j = 0
            # print(f"iiiiii {i}, {j}")
            # for i1 in range(image.shape[0]):
            #     for i2 in range(image.shape[1]):
            #             if image[i1][i2][0] == 0 and image[i1][i2][0] == image[i1][i2][1] and image[i1][i2][0] == image[i1][i2][2]:
            #                 d += 1
            # if d / (30 * 30 * 3) < 0.5:
            #     i += 1
            # x, y = (i * 30, j * 30)  # 图像叠加位置
            # i += 1
            # W1, H1 = image.shape[1::-1]
            # W2, H2 = patch.shape[1::-1]
            # if (x + W2) > W1: x = W1 - W2
            # if (y + H2) > H1: y = H1 - H2
            # print(W1, H1, W2, H2, x, y)
            # imgROI = image[y:y + H2, x:x + W2]  # 从背景图像裁剪出叠加区域图像
            b = i
            c = j
            ff = 0
            for j in range(c, 21):
                for i in range(b, 21):
                    x, y = (i * 112, j * 112)  # 图像叠加位置
                    # i += 1
                    W1, H1 = image.shape[1::-1]
                    W2, H2 = patch.shape[1::-1]
                    if (x + W2) > W1: x = W1 - W2
                    if (y + H2) > H1: y = H1 - H2
                    imgROI = image[y:y + H2, x:x + W2]  # 从背景图像裁剪出叠加区域图像
                    d = 0
                    for i1 in range(imgROI.shape[0]):
                        for i2 in range(imgROI.shape[1]):
                            if imgROI[i1][i2][0] == 117 and imgROI[i1][i2][0] == imgROI[i1][i2][1] and imgROI[i1][i2][0] == imgROI[i1][i2][2]:
                                d += 1
                    print(f"{d} / {(imgROI.shape[0] * imgROI.shape[1])} = {d / (imgROI.shape[0] * imgROI.shape[1])}")
                    if d / (imgROI.shape[0] * imgROI.shape[1]) >= 0.5:
                        # print(f"{i}, {j}")
                        # print(d)
                        ff = 1
                        break
                    if d / (imgROI.shape[0] * imgROI.shape[1]) < 0.5 and b == 20:
                        b = 0
                if ff == 1:
                    break
            # print(f"ddddd{i}, {j}")
            x, y = (i * 112, j * 112)  # 图像叠加位置
            i += 1
            W1, H1 = image.shape[1::-1]
            W2, H2 = patch.shape[1::-1]
            if (x + W2) > W1: x = W1 - W2
            if (y + H2) > H1: y = H1 - H2
            print(W1, H1, W2, H2, x, y)
            imgROI = image[y:y + H2, x:x + W2]  # 从背景图像裁剪出叠加区域图像

            img2Gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)  # img2: 转换为灰度图像
            # cv2.imshow(patch_file, img2Gray)
            # cv2.waitKey()
            ret, mask = cv2.threshold(img2Gray, 0, 255, cv2.THRESH_BINARY)  # 转换为二值图像，生成遮罩，LOGO 区域黑色遮盖
            # maskInv = mask
            # cv2.imshow(patch_file, mask)
            # cv2.waitKey()
            maskInv = cv2.bitwise_not(mask)  # 按位非(黑白转置)，生成逆遮罩，LOGO 区域白色开窗，LOGO 以外区域黑色
            # cv2.imshow(patch_file, maskInv)
            # cv2.waitKey()


            # mask 黑色遮盖区域输出为黑色，mask 白色开窗区域与运算（原图像素不变）
            img1Bg = cv2.bitwise_and(imgROI, imgROI, mask=maskInv)  # 生成背景，imgROI 的遮罩区域输出黑色
            img2Fg = cv2.bitwise_and(patch, patch, mask=mask)  # 生成前景，LOGO 的逆遮罩区域输出黑色

            # img1Bg = cv2.bitwise_or(imgROI, imgROI, mask=mask)  # 生成背景，与 cv2.bitwise_and 效果相同
            # img2Fg = cv2.bitwise_or(patch, patch, mask=maskInv)  # 生成前景，与 cv2.bitwise_and 效果相同
            # img1Bg = cv2.add(imgROI, np.zeros(np.shape(patch), dtype=np.uint8), mask=mask)  # 生成背景，与 cv2.bitwise 效果相同
            # img2Fg = cv2.add(patch, np.zeros(np.shape(patch), dtype=np.uint8), mask=maskInv)  # 生成背景，与 cv2.bitwise 效果相同
            imgROIAdd = cv2.add(img1Bg, img2Fg)  # 前景与背景合成，得到裁剪部分的叠加图像
            imgAdd = image.copy()
            imgAdd[y:y + H2, x:x + W2] = imgROIAdd  # 用叠加图像替换背景图像中的叠加位置，得到叠加 Logo 合成图像
            image = imgAdd
    image = cv2.resize(image, (600, 600))
    # with tf.io.gfile.Gfile(save_image_root,'wb') as file:
    #     file.write(image)
    cv2.imshow(patch_file, image)
    cv2.waitKey()




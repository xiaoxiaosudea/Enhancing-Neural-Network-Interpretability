import os

import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk
import cv2
import susu_ckpt_to_pb_and_predict

UNIT = 40   # pixels
MAZE_H = 1  # grid height
MAZE_W = 10  # grid width


class optim(tk.Tk, object):
    def __init__(self):
        super(optim, self).__init__()
        self.action_space = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.title('optim')
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
        self._build_maze()
        self.image_path = "D:\PaperCode\ACE-1\susu_new\ImageNet\\verify\wolf_in_snow_back\\timber_wolf\\1\saved\\0001.png"
        self.image = cv2.imread("D:\PaperCode\ACE-1\susu_new\ImageNet\\verify\wolf_in_snow_back\\timber_wolf\\1\saved\\0001.png")
        self.concept_path ="D:\PaperCode\ACE-1\susu_new\ImageNet\\verify\wolf_in_snow_back\\timber_wolf\\1\\add"
        self.save_dic = "D:\PaperCode\ACE-1\susu_new\ImageNet\\verify\wolf_in_snow_back\\timber_wolf\\1\DQN-10"
        self.i = 0
        self.j = 0
        self.real_id = 0

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                width=MAZE_W * UNIT)
        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        # hell
        # hell2_center = origin + np.array([UNIT, UNIT * 2])
        # self.hell2 = self.canvas.create_rectangle(
        #     hell2_center[0] - 15, hell2_center[1] - 15,
        #     hell2_center[0] + 15, hell2_center[1] + 15,
        #     fill='black')

        # create oval
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        a = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.rect = self.canvas.create_rectangle(
            a * 40 + 5, origin[1] - 15,
            (a + 1) * 40 - 5, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        del self.image
        self.i = 0
        self.j = 0
        self.image = cv2.imread("D:\PaperCode\ACE-2\susu_new\ACE\tench_fore_concepts_patch\images\\0001.png")
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        a = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.rect = self.canvas.create_rectangle(
            a * 40 + 5, origin[1] - 15,
            (a + 1) * 40 - 5, origin[1] + 15,
            fill='red')
        # return observation
        return (np.array(self.canvas.coords(self.rect)[:2]))/(MAZE_H*UNIT)



    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        base_action[1] = 0
        # print(s)
        if action == 0:   # 0
            if action < s[0] // 40:
                base_action[0] -= ((s[0] // 40) - 0) * UNIT
            else:
                base_action[0] += (0-(s[0] // 40)) * UNIT
            # if s[1] > UNIT:
            #     base_action[1] -= UNIT
        elif action == 1:   # 1
            # if action != s[0] / 40+ 1:
            if action < s[0] // 40:
                base_action[0] -= ((s[0] // 40) - 1) * UNIT
            else:
                base_action[0] += (1-(s[0] // 40)) * UNIT
            # if s[1] < (MAZE_H - 1) * UNIT:
            #     base_action[1] += UNIT
        elif action == 2:   # 2
            # if action != s[0] / 40+ 1:
            if action < s[0] // 40:
                base_action[0] -= ((s[0] // 40) - 2) * UNIT
            else:
                base_action[0] += (2 - (s[0] // 40)) * UNIT
            # if s[0] < (MAZE_W - 1) * UNIT:
            #     base_action[0] += UNIT
        elif action == 3:   # 3
            # if action != s[0] / 40+ 1:
            if action < s[0] // 40:
                base_action[0] -= ((s[0] // 40) - 3) * UNIT
            else:
                base_action[0] += (3 - (s[0] // 40)) * UNIT
            # if s[0] > UNIT:
            #     base_action[0] -= UNIT
        elif action == 4:  # 4
            # if action != s[0] / 40 + 1:
            if action < s[0] // 40:
                base_action[0] -= ((s[0] // 40) - 4) * UNIT
            else:
                base_action[0] += (4 - (s[0] // 40)) * UNIT
        elif action == 5:  # 5
            # if action != s[0] / 40 + 1:
            if action < s[0] // 40:
                base_action[0] -= ((s[0] // 40) - 5) * UNIT
            else:
                base_action[0] += (5 - (s[0] // 40)) * UNIT
        elif action == 6:  # 6
            # if action != s[0] / 40 + 1:
            if action < s[0] // 40:
                base_action[0] -= ((s[0] // 40) - 6) * UNIT
            else:
                base_action[0] += (6 - (s[0] // 40)) * UNIT
        elif action == 7:  # 7
            # if action != s[0] / 40 + 1:
            if action < s[0] // 40:
                base_action[0] -= ((s[0] // 40) - 7) * UNIT
            else:
                base_action[0] += (7 - (s[0] // 40)) * UNIT
        elif action == 8:  # 8
            # if action != s[0] / 40 + 1:
            if action < s[0] // 40:
                base_action[0] -= ((s[0] // 40) - 8) * UNIT
            else:
                base_action[0] += (8 - (s[0] // 40)) * UNIT
        elif action == 9:  # 9
            # if action != s[0] / 40 + 1:
            if action < s[0] // 40:
                base_action[0] -= ((s[0] // 40) - 9) * UNIT
            else:
                base_action[0] += (9 - (s[0] // 40)) * UNIT
        # elif action == 10:  # 4
        #     # if action != s[0] / 40 + 1:
        #     if action < s[0] // 40:
        #         base_action[0] -= ((s[0] // 40) - 10) * UNIT
        #     else:
        #         base_action[0] += (10 - (s[0] // 40)) * UNIT


        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        next_coords = self.canvas.coords(self.rect)  # next state

        # image送入预测网络
        class_id_start = susu_ckpt_to_pb_and_predict.freeze_graph_test_1("D:\PaperCode\ACE-2\susu_new\Deep_Q_Network\susu_V1_0.9766.pb", self.image_path)
        # cv2.imshow("patch_file", self.image)
        # cv2.waitKey()
        self.image = cv2.resize(self.image, (2240, 2240))
        patches_root = os.path.join(self.concept_path, f"concept_{action+1}")
        patches = os.listdir(patches_root)
        for patch_file in patches:
            patch_path = os.path.join(patches_root, patch_file)
            patch = cv2.imread(patch_path)
            patch = cv2.resize(patch, (84, 84))

            # d = 0
            if self.i * 56 > self.image.shape[1]:
                self.i = 0
                self.j += 1
            if self.j * 56 > self.image.shape[0]:
                self. i = 0
                self. j = 0
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
            b = self.i
            c = self.j
            ff = 0
            for self.j in range(c, 31):
                for self.i in range(b, 31):
                    # print(f"{self.i}, {self.j}")
                    x, y = (self.i * 84, self.j * 84)  # 图像叠加位置
                    # i += 1
                    W1, H1 = self.image.shape[1::-1]
                    W2, H2 = patch.shape[1::-1]
                    if (x + W2) > W1: x = W1 - W2
                    if (y + H2) > H1: y = H1 - H2
                    imgROI = self.image[y:y + H2, x:x + W2]  # 从背景图像裁剪出叠加区域图像
                    d = 0
                    for i1 in range(imgROI.shape[0]):
                        for i2 in range(imgROI.shape[1]):
                            if imgROI[i1][i2][0] == 117 and imgROI[i1][i2][0] == imgROI[i1][i2][1] and imgROI[i1][i2][0] == imgROI[i1][i2][2]:
                                d += 1
                                # print(d)
                    # print(f"{d} / {(imgROI.shape[0] * imgROI.shape[1])} = {d / (imgROI.shape[0] * imgROI.shape[1])}")
                    if d / (imgROI.shape[0] * imgROI.shape[1]) >= 0.5:
                        # print(f"{self.i}, {self.j}")
                        # print(d)
                        ff = 1
                        break
                    if d / (imgROI.shape[0] * imgROI.shape[1]) < 0.5 and b == 20:
                        b = 0
                if ff == 1:
                    break
            # print(f"ddddd{i}, {j}")
            x, y = (self.i * 84, self.j * 84)  # 图像叠加位置
            self.i += 1
            W1, H1 = self.image.shape[1::-1]
            W2, H2 = patch.shape[1::-1]
            if (x + W2) > W1: x = W1 - W2
            if (y + H2) > H1: y = H1 - H2
            # print(W1, H1, W2, H2, x, y)
            imgROI = self.image[y:y + H2, x:x + W2]  # 从背景图像裁剪出叠加区域图像

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
            imgAdd = self.image.copy()
            imgAdd[y:y + H2, x:x + W2] = imgROIAdd  # 用叠加图像替换背景图像中的叠加位置，得到叠加 Logo 合成图像
            self.image = imgAdd
        self.image = cv2.resize(self.image, (600, 600))

        # cv2.imshow(patch_file, image)
        # cv2.waitKey()
        sy_image_path = self.image_path.split('\\')[-1]
        sy_image_path = os.path.join(self.save_dic, sy_image_path)
        cv2.imwrite(sy_image_path, self.image)
        # 送进预测网络
        class_id_final = susu_ckpt_to_pb_and_predict.freeze_graph_test_1("D:\PaperCode\ACE-2\susu_new\Deep_Q_Network\susu_V1_0.9766.pb", sy_image_path)


        # reward function
        if class_id_start != self.real_id and class_id_final == self.real_id:
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        # if next_coords == [125.0, 5.0, 155.0, 35.0]:
        #     reward = 1
        #     done = True
        # elif next_coords == [165, 5, 195, 35]:
        #     reward = -1
        #     done = True
        # else:
        #     reward = 0
        #     done = False
        s_ = (np.array(next_coords[:2]))/(MAZE_H*UNIT)
        return s_, reward, done

    def render(self):
        # time.sleep(0.1)
        self.update()


if __name__ == '__main__':
    env = optim()
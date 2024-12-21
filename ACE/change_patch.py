import cv2
import PIL.Image
import numpy as np


def main(png_path_1, png_path_2, png_path_change):
    png_arrayy_1 = PIL.Image.open(png_path_1).convert("RGB")
    png_arrayy_2 = PIL.Image.open(png_path_2).convert("RGB")
    png_arrayy_change =PIL.Image.open(png_path_change).convert("RGB")
    # png_arrayy_1 = cv2.imread(png_path_1)
    # png_arrayy_2 = cv2.imread(png_path_2)
    # png_arrayy_change = cv2.imread(png_path_change)
    png_arrayy_1 = np.array(png_arrayy_1)
    png_arrayy_2 = np.array(png_arrayy_2)
    png_arrayy_change = np.array(png_arrayy_change)
    x, y, z = png_arrayy_1.shape
    for i in range(x):
        for j in range(y):
            for m in range(z):
                if png_arrayy_change[i][j][m] == 117:
                    png_arrayy_change[i][j][m] = 0
    for i in range(x):
        for j in range(y):
            for m in range(z):
                if png_arrayy_2[i][j][m] != 117:
                    png_arrayy_1[i][j][m] = 0
    cv2.imwrite("C:\\Users\\Administrator\\software\\Project\\susu_use\\picture\\0036_change.png", png_arrayy_1)
    png_arrayy_1 = PIL.Image.fromarray(png_arrayy_1)
    png_arrayy_change = PIL.Image.fromarray(png_arrayy_change)
    width = png_arrayy_change.size[0]  # 获取宽度
    height = png_arrayy_change.size[1]  # 获取高度
    png_arrayy_change = png_arrayy_change.resize((int(width * 0.3), int(height * 0.3)), PIL.Image.ANTIALIAS)
    png_arrayy_1.paste(png_arrayy_change, (155, 0))
    png_arrayy_1.save("C:\\Users\\Administrator\\software\\Project\\susu_use\\picture\\0036_change_new.png")
    # for i in range(x):
    #     for j in range(y):
    #         for m in range(z):
    #             if png_arrayy_1[i][j][m] == -1:
    #                 break




if __name__ == "__main__":
    png_path_1 = "C:\\Users\\Administrator\\software\\Project\\susu_use\\picture\\0036.png"
    png_path_2 = "C:\\Users\\Administrator\\software\\Project\\susu_use\\picture\\0036.png"
    png_path_change = "C:\\Users\\Administrator\\software\\Project\\susu_use\\picture\\030_26.png"
    main(png_path_1, png_path_2, png_path_change)

import pickle
import os
import numpy as np


def main(pkl_dis, save_path):
    pkl_name_all = os.listdir(pkl_dis)
    step = 0
    average_cv = None
    for name in pkl_name_all:
        save_name = name.split('_1.pkl')[0] + "_average.pkl"
        # save_name = name.split('_1')[0]+"average"
        if step == 20:
            step = 0
        # if step == 19:
            save_name = name.split('_1.pkl')[0] + "_average.pkl"
        if step == 0:
            with open(os.path.join(pkl_dis, name), 'rb') as f:
                c = pickle.load(f)
            average_cv = np.zeros_like(c)
        pkl_path = os.path.join(pkl_dis, name)
        with open(pkl_path, 'rb') as f:
             c = pickle.load(f)
             average_cv = np.add(average_cv, c)
        if step == 19 and step != 0:
            average_cv = average_cv / 20
            with open(os.path.join(save_path, save_name), 'wb') as F:
                pickle.dump(average_cv, F)
            print(average_cv.shape)
        step += 1



if __name__ == '__main__':
    # root_E_B = 'D:\PaperCode\ACE-1\susu_new\CAV\Eskimo_dog_back'
    # save_path_E_B = "D:\PaperCode\ACE-1\susu_new\CAV_end\Eskimo_dog_back"
    # main(root_E_B, save_path_E_B)
    #
    # root_E_F = 'D:\PaperCode\ACE-1\susu_new\CAV\Eskimo_dog_fore'
    # save_path_E_F = "D:\PaperCode\ACE-1\susu_new\CAV_end\Eskimo_dog_fore"
    # main(root_E_F, save_path_E_F)

    root_T_B = 'D:\PaperCode\ACE-1\susu_new\CAV\\timber_wolf_back'
    save_path_T_B = 'D:\PaperCode\ACE-1\susu_new\CAV_end\\timber_wolf_back'
    main(root_T_B, save_path_T_B)

    root_T_F = 'D:\PaperCode\ACE-1\susu_new\CAV\\timber_wolf_fore'
    save_path_T_F = 'D:\PaperCode\ACE-1\susu_new\CAV_end\\timber_wolf_fore'
    main(root_T_F, save_path_T_F)

    # root_W_B = 'D:\PaperCode\ACE-1\susu_new\CAV\white_wolf_back'
    # save_path_W_B = 'D:\PaperCode\ACE-1\susu_new\CAV_end\white_wolf_back'
    # main(root_W_B, save_path_W_B)
    #
    # root_W_F = 'D:\PaperCode\ACE-1\susu_new\CAV\white_wolf_fore'
    # save_path_W_F = 'D:\PaperCode\ACE-1\susu_new\CAV_end\white_wolf_fore'
    # main(root_W_F, save_path_W_F)

    # root = 'D:\PaperCode\ACE-1\susu_new\CAV\husky_in_grass_back'
    # save_path = 'D:\PaperCode\ACE-1\susu_new\CAV_end\husky_in_grass_back'
    # main(root, save_path)
    #
    # root = 'D:\PaperCode\ACE-1\susu_new\CAV\husky_in_grass_fore'
    # save_path = 'D:\PaperCode\ACE-1\susu_new\CAV_end\husky_in_grass_fore'
    # main(root, save_path)
    #
    # root = 'D:\PaperCode\ACE-1\susu_new\CAV\husky_in_snow_back'
    # save_path = 'D:\PaperCode\ACE-1\susu_new\CAV_end\husky_in_snow_back'
    # main(root, save_path)
    #
    # root = 'D:\PaperCode\ACE-1\susu_new\CAV\husky_in_snow_fore'
    # save_path = 'D:\PaperCode\ACE-1\susu_new\CAV_end\husky_in_snow_fore'
    # main(root, save_path)
    #
    # root = 'D:\PaperCode\ACE-1\susu_new\CAV\wolf_in_grass_back'
    # save_path = 'D:\PaperCode\ACE-1\susu_new\CAV_end\wolf_in_grass_back'
    # main(root, save_path)
    #
    # root = 'D:\PaperCode\ACE-1\susu_new\CAV\wolf_in_grass_fore'
    # save_path = 'D:\PaperCode\ACE-1\susu_new\CAV_end\wolf_in_grass_fore'
    # main(root, save_path)
    #
    # root = 'D:\PaperCode\ACE-1\susu_new\CAV\wolf_in_snow_back'
    # save_path = 'D:\PaperCode\ACE-1\susu_new\CAV_end\wolf_in_snow_back'
    # main(root, save_path)
    #
    # root = 'D:\PaperCode\ACE-1\susu_new\CAV\wolf_in_snow_fore'
    # save_path = 'D:\PaperCode\ACE-1\susu_new\CAV_end\wolf_in_snow_fore'
    # main(root, save_path)
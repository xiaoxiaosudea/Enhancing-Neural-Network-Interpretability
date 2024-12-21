import pickle
import os


def main(pkl_dis):
    pkl_name = os.listdir(pkl_dis)
    for name in pkl_name:
        pkl_path = os.path.join(pkl_dis, name)
        with open(pkl_path, 'rb') as f:
            c = pickle.load(f)
            print(c)
            print(c.shape)


if __name__ == '__main__':
    root = 'D:\PaperCode\ACE-1\susu_new\CAV\Eskimo_dog_back'
    main(root)
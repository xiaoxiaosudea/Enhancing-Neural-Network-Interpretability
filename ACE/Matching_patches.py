import os
import PIL.Image

if __name__ == '__main__':
    picture_root = "D:\PaperCode\ACE-1\susu_new\ACE\\timber_wolf_fore_concepts_patch\images"  # 原图文件夹
    patches_root = "D:\PaperCode\ACE-1\susu_new\ACE\\timber_wolf_fore_concepts_patch"  # patch的大文件夹
    save_path = "D:\PaperCode\ACE-1\susu_new\After combing\\timber_wolf_fore"

    ori_png = os.listdir(picture_root)
    patch_dic = os.listdir(patches_root)
    patch_dic.remove("images")
    for png in ori_png:
        name_id = int(png.split(".png")[0]) - 1
        for patch in patch_dic:
            patch_dic_path = os.path.join(patches_root, patch)
            patches_names = os.listdir(patch_dic_path)
            for patch_1 in patches_names:
                if int(patch_1.split("_")[1].split(".png")[0]) == name_id:
                    image = PIL.Image.open(os.path.join(patch_dic_path, patch_1))
                    if not os.path.exists(os.path.join(save_path, png.split(".png")[0])):
                        os.makedirs(os.path.join(save_path, png.split(".png")[0]))
                    if not os.path.exists(os.path.join(save_path, png.split(".png")[0], patch)):
                        os.makedirs(os.path.join(save_path, png.split(".png")[0], patch))
                    image.save(os.path.join(save_path, png.split(".png")[0], patch, patch_1))
        image = PIL.Image.open(os.path.join(picture_root, png))
        # if not os.path.exists()
        image.save(os.path.join(save_path, png.split(".png")[0], png))

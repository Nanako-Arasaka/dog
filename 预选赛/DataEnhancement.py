import os
import numpy as np
from PIL import Image
import shutil
import matplotlib.pyplot as plt
import albumentations as A

def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False

if __name__ == "__main__":
    IMG_DIR = r"C:\Users\AT-austin\Desktop\The Dog\预选赛\data\train\low"  ### 原始数据集图像的路径
    AUG_IMG_DIR = r"C:\Users\AT-austin\Desktop\The Dog\预选赛\data\train_enhanced\low"### 数据增强后图片的保存路径

    try:
        shutil.rmtree(AUG_IMG_DIR)
    except FileNotFoundError as e:
        pass

    mkdir(AUG_IMG_DIR)

    AUGLOOP = 2# 每张影像增强的数量

    # 影像增强
    transform = A.Compose([
        A.VerticalFlip(p=0.5),  # vertically flip 50% of all images
        A.HorizontalFlip(p=0.5),  # 镜像
        A.RandomBrightnessContrast(brightness_limit=(0.1, 0.4), contrast_limit=0, p=1.0),  # change brightness
        A.GaussianBlur(blur_limit=(0, 7), p=1.0),  # apply Gaussian blur
        A.Affine(
            translate_percent={"x": 0.05, "y": 0.05},
            scale=(0.8, 0.95),
            rotate=(-30, 30),
            p=1.0
        )  # translate and scale
    ])

    for root, sub_folders, files in os.walk(IMG_DIR):
        for name in files:
            print(name)
            shutil.copy(os.path.join(IMG_DIR, name), AUG_IMG_DIR)

            for epoch in range(AUGLOOP):
                img = Image.open(os.path.join(IMG_DIR, name))
                img = np.asarray(img)
    
                image_aug = transform(image=img)['image']
                path = os.path.join(AUG_IMG_DIR, name[:-4] + str("_%06d" % (epoch + 1)) + '.jpg')
                Image.fromarray(image_aug).save(path)
    
                print(name[:-4] + str("_%06d" % (epoch + 1)) + '.jpg')
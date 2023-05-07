import os
import shutil
from random import Random


def auto_mkdir(path):
    if os.path.exists(path):
        if os.listdir(images_path):
            raise "路径%s已经存在，请先删除" % images_path
    else:
        os.mkdir(path)


if __name__ == '__main__':
    train_size = 0.7
    rd = Random(42)
    # 检测路径是否存在
    base_path = '.\\african-wildlife'
    target_base_path = '.\\african-wildlife-dataset'
    assert os.path.exists(base_path), "路径%s不存在" % base_path
    paths = ['buffalo', 'elephant', 'rhino', 'zebra']
    paths = [os.path.join(base_path, p) for p in paths]
    for p in paths:
        assert os.path.exists(p), "路径%s不存在" % p
    # 创建基本路径，要求路径不存在或者为空
    images_path = os.path.join(target_base_path, 'images')
    labels_path = os.path.join(target_base_path, 'labels')
    auto_mkdir(target_base_path)
    auto_mkdir(images_path)
    auto_mkdir(labels_path)
    for s in ['train', 'val']:
        auto_mkdir(os.path.join(images_path, s))
        auto_mkdir(os.path.join(labels_path, s))
    # 分割训练集和验证集
    train_paths = []
    val_paths = []
    for path in paths:
        files = os.listdir(path)
        imgs_files = filter(lambda x: x[-4:] == '.jpg', files)
        imgs_files = [os.path.join(path, p) for p in imgs_files]
        rd.shuffle(imgs_files)
        sp = int(len(imgs_files) * train_size)
        train_paths.extend(imgs_files[:sp])
        val_paths.extend(imgs_files[sp + 1:])
    rd.shuffle(train_paths)
    rd.shuffle(val_paths)
    # 移动文件
    idx = 0
    for path in train_paths:
        file_name = os.path.join('train', str(idx).zfill(12))
        target_img_path = os.path.join(images_path, file_name + ".jpg")
        target_lbl_path = os.path.join(labels_path, file_name + ".txt")
        shutil.copy(path, target_img_path)
        shutil.copy(path.replace('.jpg', '.txt'), target_lbl_path)
        idx += 1
    for path in val_paths:
        file_name = os.path.join('val', str(idx).zfill(12))
        target_img_path = os.path.join(images_path, file_name + ".jpg")
        target_lbl_path = os.path.join(labels_path, file_name + ".txt")
        shutil.copy(path, target_img_path)
        shutil.copy(path.replace('.jpg', '.txt'), target_lbl_path)
        idx += 1

# 生成包含路径和类别信息的txt文件
import glob
import os

base_path = './classify_dataset/images'
names = {"tin": 0, "bottle": 1, "box": 2, "bag": 3, "bucket": 4}
for type in os.listdir(base_path):
    if type == 'train':
        with open('classify_dataset/images/train/train.txt', mode='w+', encoding='utf-8') as file:
            for category in os.listdir(os.path.join(base_path, type)):
                part_paths = glob.glob(os.path.join(base_path, type, category, '*'))
                part_content = [part_path + ' ' + str(names[category]) + '\n' for part_path in part_paths]
                f = file.writelines(part_content)
    if type == 'val':
        with open('classify_dataset/images/val/val.txt', mode='w+', encoding='utf-8') as file:
            for category in os.listdir(os.path.join(base_path, type)):
                part_paths = glob.glob(os.path.join(base_path, type, category, '*'))
                part_content = [part_path + ' ' + str(names[category]) + '\n' for part_path in part_paths]
                f = file.writelines(part_content)
# 直方图均衡化
# 增强清晰度
import os
import cv2
from tqdm import tqdm
def opt_img(img_path):
    img = cv2.imread(img_path, 0)
    clahe = cv2.createCLAHE(tileGridSize=(3, 3))
    # 自适应直方图均衡化
    dst1 = clahe.apply(img)
    cv2.imwrite(img_path, dst1)

# 原照片同路径，替代原有模糊照片
pic_path_folder = r'D:\pythonprofessor\Yolov5\BoneAge_classify_dataset\arthrosis_ori\arthrosis_ori\Ulna\12'
if __name__ == '__main__':
    for pic_folder in tqdm(os.listdir(pic_path_folder)):
        data_path = os.path.join(pic_path_folder, pic_folder)
        # 去雾
        opt_img(data_path)
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from BoneAge_tool import common

class Yolov5Dataset(Dataset):
    def __init__(self, joint_type, mode_type):
        # 定义一个列表用于存放图片路径以及类别
        self.dataset = []
        # 图片主干目录
        base_path = r'D:\pythonprofessor\Yolov5\BoneAge_classify_dataset\arthrosis_ori\arthrosis_ori'
        # 关节点的类型
        arthrosis = common.arthrosis[joint_type][0]
        # 训练还是测试
        mode_train_or_val = mode_type
        # 训练模式下每张图片的路径
        if mode_train_or_val == 'train':
            self.arthrosis_path = os.path.join(base_path, arthrosis, 'train.txt')
        # 验证模式下每张图片的路径
        if mode_train_or_val == 'val':
            self.arthrosis_path = os.path.join(base_path, arthrosis, 'val.txt')
        # 打开文件
        with open(self.arthrosis_path, 'r', encoding='utf-8') as file:
        # 打开之后进行多行文件的读取
            f = file.readlines()
        # 遍历读取文件的每一行
            for line in f:
        # 对每一行的数据进行处理
        # 获取分类的数据
                cls = line.strip().split()[-1]
        # 获取路径信息
                path = line.strip().split()[0]
        # 将路径与类别信息打包放置在一起
                self.dataset.append([cls, path])

    def __len__(self):
        # 此处相当于统计长度，即训练图片总的个数
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        一般用于数据的处理，利用Dataloader将之前打包好的数据集解包，数据处理好之后就返回给Dataloader
        （如果是多个数据的话比如return label, cls是会自己打包成元组）
        """
        """
        这里可以认为，(Dataloader将初始化的数据这里是self.dataset打包接受了。
        而在getitem就接受Dataloader传过来的self.dataset数据,并自动以idx索引控制进行遍历处理)
        """
        # 获取self.dataset里面存放的类别和该图片的路径
        cls, path = self.dataset[idx]
        # 通过地址取出图片(只是取出图片，并不是显示)
        image = Image.open(path)
        # 将图片转为正方形
        square_image = common.trans_square(image)
        # 将图片转为灰度图
        gray_image = square_image.convert('L')
        # 放大图片到224
        amplify_image = common.data_transforms(gray_image)
        # 将类别的标签张量化
        cls_tensor = torch.tensor(int(cls))
        # 返回图像以及分类标签的张量化
        return cls_tensor, amplify_image

if __name__ == '__main__':
    # 对象实例化，对于类来说需要给其赋值
    yolov5 = Yolov5Dataset(joint_type="DIPThird", mode_type='train')
    # 打印的结果应该是返回数据集的长度，len里面必须是实例名（也可能是数据集名）才能得到魔法len的返回长度统计
    print(len(yolov5))
    # 通过对实例用[]来调用getitem方法，此处给idx赋值为0，来进行测试
    test_dataset = yolov5[0]
    # 打印返回该实例处于[idx]的值，如果getitem有返回值的话
    print(test_dataset)
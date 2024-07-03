# import os.path
# import torch
# from main_Dataset import Yolov5Dataset
# import tqdm
# from torch import nn
# from torchvision import models
# from torch import optim
# from torch.utils.data import DataLoader
# from BoneAge_tool import common
# from PIL import Image
# from torchvision.transforms import ToTensor
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# class classifier():
#     def __init__(self, arthrosis_name, mode_name):
#         super(classifier, self).__init__()
#         # 通过输入的关节名，确定保存的权重名字
#         self.weight_path = f'./weights/{arthrosis_name}.pt'
#         # 调用common文件，输入各关节对应的类别数量
#         cls = common.arthrosis[arthrosis_name][1]
#         self.model = self.new_model(cls).to(device)
#         if mode_name == 'train':
#             self.train_datasets = DataLoader(Yolov5Dataset(arthrosis_name, 'train'), batch_size=32, shuffle=True, num_workers=16)
#         if mode_name == 'val':
#             self.test_datasets = DataLoader(Yolov5Dataset(arthrosis_name, 'val'), batch_size=32, shuffle=True, num_workers=16)
#         if os.path.exists(self.weight_path):
#             self.model.load_state_dict(torch.load(self.weight_path, map_location=torch.device('cpu')))
#         self.opt = optim.Adam(params=self.model.parameters())
#         self.loss_fn = nn.CrossEntropyLoss()
#
#         # 传入的cls只是个期待值，用来确定输出的类别数量
#     def new_model(self, cls):
#         net = models.resnet18()
#         net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         net.fc = nn.Linear(512, cls)
#         # 返回的是修改后的网络，而不是分类数
#         return net.to(device)
#
#     def train_open(self, epochs, best_acc):
#         sum_loss = 0
#         sum_acc = 0
#         self.model.train()
#         for cls, image in tqdm.tqdm(self.train_datasets, desc='训练中：'):
#             cls, image = cls.to(device), image.to(device)
#             # 前向传播
#             model_cls = self.model(image).to(device)
#             acc = torch.mean(torch.eq(torch.argmax(model_cls), cls))
#             # 实际值在后，计算损失
#             loss = self.loss_fn(model_cls, cls)
#             sum_acc += acc.item()
#             # 计算损失和
#             sum_loss += loss.item()
#             # 梯度清零
#             self.opt.zero_grad()
#             # 反向传播
#             loss.backward()
#             # 梯度更新
#             self.opt.step()
#         ave_acc = sum_acc / len(self.train_datasets)
#         ave_loss = sum_loss / len(self.train_datasets)
#         print(f'第{epochs}轮,训练的平均损失为:{ave_loss}, 训练的平均准确度为：{ave_acc}')
#         if ave_acc > best_acc:
#             best_acc = ave_acc
#             torch.save(self.model.state_dict(), self.weight_path)
#         return best_acc
#
#
#     def test(self, epochs):
#         sum_loss = 0
#         sum_acc = 0
#         self.model.eval()
#         for cls, image in tqdm.tqdm(self.test_datasets, desc='测试中：'):
#             cls, image = cls.to(device), image.to(device)
#             # 前向传播
#             model_cls = self.model(image).to(device)
#             acc = torch.mean(torch.eq(torch.argmax(model_cls), cls))
#             # 实际值在后，计算损失
#             loss = self.loss_fn(model_cls, cls)
#             sum_acc += acc.item()
#             # 计算损失和
#             sum_loss += loss.item()
#         ave_acc = sum_acc / len(self.test_datasets)
#         ave_loss = sum_loss / len(self.test_datasets)
#         print(f'第{epochs}轮,测试的平均损失为:{ave_loss},测试的平均准确度为：{ave_acc}')
#
#     def detect(self, part_image, joint_name, sex):
#         part_image = part_image.unsqueeze(0)
#         self.model.eval()
#         if joint_name == 'DIPFifth' or joint_name == 'DIPThird':
#             # 需要debug检测一下grade是什么值用argmax，再看最高是多少，需不需要grade-1
#             old_grade = self.model(part_image)
#             grade = old_grade.shape[1]
#             if sex == 'girl':
#                 score = common.SCORE[sex][joint_name][grade-1]
#                 return score
#             elif sex == 'boy':
#                 score = common.SCORE[sex][joint_name][grade-1]
#                 return score
#         elif joint_name == 'DIPFirst':
#             old_grade = self.model(part_image)
#             grade = old_grade.shape[1]
#             if sex == 'girl':
#                 score = common.SCORE[sex][joint_name][grade-1]
#                 return score
#             elif sex == 'boy':
#                 score = common.SCORE[sex][joint_name][grade-1]
#                 return score
#         elif joint_name == 'MCPFifth' or joint_name == 'MCPThird':
#             old_grade = self.model(part_image)
#             grade = old_grade.shape[1]
#             if sex == 'girl':
#                 score = common.SCORE[sex][joint_name][grade-1]
#                 return score
#             elif sex == 'boy':
#                 score = common.SCORE[sex][joint_name][grade-1]
#                 return score
#         elif joint_name == 'MCPFirst':
#             old_grade = self.model(part_image)
#             grade = old_grade.shape[1]
#             if sex == 'girl':
#                 score = common.SCORE[sex][joint_name][grade-1]
#                 return score
#             elif sex == 'boy':
#                 score = common.SCORE[sex][joint_name][grade-1]
#                 return score
#         elif joint_name == 'MIPFifth' or joint_name == 'MIPThird':
#             old_grade = self.model(part_image)
#             grade = old_grade.shape[1]
#             if sex == 'girl':
#                 score = common.SCORE[sex][joint_name][grade-1]
#                 return score
#             elif sex == 'boy':
#                 score = common.SCORE[sex][joint_name][grade-1]
#                 return score
#         elif joint_name == 'PIPFifth' or joint_name == 'PIPThird':
#             old_grade = self.model(part_image)
#             grade = old_grade.shape[1]
#             if sex == 'girl':
#                 score = common.SCORE[sex][joint_name][grade-1]
#                 return score
#             elif sex == 'boy':
#                 score = common.SCORE[sex][joint_name][grade-1]
#                 return score
#         elif joint_name == 'PIPFirst':
#             old_grade = self.model(part_image)
#             grade = old_grade.shape[1]
#             if sex == 'girl':
#                 score = common.SCORE[sex][joint_name][grade-1]
#                 return score
#             elif sex == 'boy':
#                 score = common.SCORE[sex][joint_name][grade-1]
#                 return score
#         elif joint_name == 'Radius':
#             old_grade = self.model(part_image)
#             grade = old_grade.shape[1]
#             if sex == 'girl':
#                 score = common.SCORE[sex][joint_name][grade-1]
#                 return score
#             elif sex == 'boy':
#                 score = common.SCORE[sex][joint_name][grade-1]
#                 return score
#         elif joint_name == 'Ulna':
#             old_grade = self.model(part_image)
#             grade = old_grade.shape[1]
#             if sex == 'girl':
#                 score = common.SCORE[sex][joint_name][grade-1]
#                 return score
#             elif sex == 'boy':
#                 score = common.SCORE[sex][joint_name][grade-1]
#                 return score
#
#
#     def run(self, part_image, joint_name, sex, choose_mode):
#         best_acc = 0
#         if choose_mode == 'train':
#             for epochs in range(100):
#                 self.train_open(epochs, best_acc)
#         elif choose_mode == 'val':
#             for epochs in range(100):
#                 self.test(epochs)
#         elif choose_mode == 'detect':
#             part_score = self.detect(part_image, joint_name, sex)
#             return part_score
#
#
# if __name__ == '__main__':
#     image_path = './img.png'
#     part_image = Image.open(image_path)
#     gray_image = part_image.convert('L')
#     to_tensor = ToTensor()
#     part_image = to_tensor(gray_image)
#     classifier = classifier('DIPFirst', 'detect')
#     run = classifier.run(part_image, 'DIPFirst', 'boy', 'detect')


# 新版
import os.path
import torch
from main_Dataset import Yolov5Dataset
import tqdm
from torch import nn
from torchvision import models
from torch import optim
from torch.utils.data import DataLoader
from BoneAge_tool import common
from PIL import Image
from torchvision.transforms import ToTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class classifier():
    def __init__(self, arthrosis_name, mode_name):
        super(classifier, self).__init__()
        # 通过输入的关节名，确定保存的权重名字
        self.weight_path = f'./weights/{arthrosis_name}.pt'
        # 调用common文件，输入各关节对应的类别数量
        cls = common.arthrosis[arthrosis_name][1]
        self.model = self.new_model(cls).to(device)
        if mode_name == 'train':
            self.train_datasets = DataLoader(Yolov5Dataset(arthrosis_name, 'train'), batch_size=32, shuffle=True, num_workers=16)
        if mode_name == 'val':
            self.test_datasets = DataLoader(Yolov5Dataset(arthrosis_name, 'val'), batch_size=32, shuffle=True, num_workers=16)
        if os.path.exists(self.weight_path):
            self.model.load_state_dict(torch.load(self.weight_path, map_location=torch.device('cpu')))
        self.opt = optim.Adam(params=self.model.parameters())
        self.loss_fn = nn.CrossEntropyLoss()


        # 传入的cls只是个期待值，用来确定输出的类别数量
    def new_model(self, cls):
        net = models.resnet18()
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        net.fc = nn.Linear(512, cls)
        # 返回的是修改后的网络，而不是分类数
        return net.to(device)

    def train_open(self, epochs):
        sum_loss = 0
        sum_acc = 0
        self.model.train()
        for cls, image in tqdm.tqdm(self.train_datasets, desc='训练中：'):
            cls, image = cls.to(device), image.to(device)
            # 前向传播
            model_cls = self.model(image).to(device)
            acc = torch.mean(torch.eq(torch.argmax(model_cls), cls))
            # 实际值在后，计算损失
            loss = self.loss_fn(model_cls, cls)
            sum_acc += acc.item()
            # 计算损失和
            sum_loss += loss.item()
            # 梯度清零
            self.opt.zero_grad()
            # 反向传播
            loss.backward()
            # 梯度更新
            self.opt.step()
        ave_acc = sum_acc / len(self.train_datasets)
        ave_loss = sum_loss / len(self.train_datasets)
        print(f'第{epochs}轮,训练的平均损失为:{ave_loss}, 训练的平均准确度为：{ave_acc}')


    def test(self, epochs, best_acc):
        sum_loss = 0
        sum_acc = 0
        self.model.eval()
        for cls, image in tqdm.tqdm(self.test_datasets, desc='测试中：'):
            cls, image = cls.to(device), image.to(device)
            # 前向传播
            model_cls = self.model(image).to(device)
            acc = torch.mean(torch.eq(torch.argmax(model_cls), cls))
            # 实际值在后，计算损失
            loss = self.loss_fn(model_cls, cls)
            sum_acc += acc.item()
            # 计算损失和
            sum_loss += loss.item()
        ave_acc = sum_acc / len(self.test_datasets)
        ave_loss = sum_loss / len(self.test_datasets)
        if ave_acc > best_acc:
            best_acc = ave_acc
            torch.save(self.model.state_dict(), self.weight_path)
        print(f'第{epochs}轮,测试的平均损失为:{ave_loss},测试的平均准确度为：{ave_acc}, 最好准确度为：{best_acc}')

    def detect(self, part_image, joint_name, sex):
        part_image = part_image.unsqueeze(0)
        self.model.eval()
        if joint_name == 'DIPFifth' or joint_name == 'DIPThird':
            # 需要debug检测一下grade是什么值用argmax，再看最高是多少，需不需要grade-1
            old_grade = self.model(part_image)
            grade = torch.argmax(old_grade[0])
            if sex == 'girl':
                score = common.SCORE[sex][joint_name][grade]
                return score
            elif sex == 'boy':
                score = common.SCORE[sex][joint_name][grade]
                return score
        elif joint_name == 'DIPFirst':
            old_grade = self.model(part_image)
            grade = torch.argmax(old_grade[0])
            if sex == 'girl':
                score = common.SCORE[sex][joint_name][grade]
                return score
            elif sex == 'boy':
                score = common.SCORE[sex][joint_name][grade]
                return score
        elif joint_name == 'MCPFifth' or joint_name == 'MCPThird':
            old_grade = self.model(part_image)
            grade = torch.argmax(old_grade[0])
            if sex == 'girl':
                score = common.SCORE[sex][joint_name][grade]
                return score
            elif sex == 'boy':
                score = common.SCORE[sex][joint_name][grade]
                return score
        elif joint_name == 'MCPFirst':
            old_grade = self.model(part_image)
            grade = torch.argmax(old_grade[0])
            if sex == 'girl':
                score = common.SCORE[sex][joint_name][grade]
                return score
            elif sex == 'boy':
                score = common.SCORE[sex][joint_name][grade]
                return score
        elif joint_name == 'MIPFifth' or joint_name == 'MIPThird':
            old_grade = self.model(part_image)
            grade = torch.argmax(old_grade[0])
            if sex == 'girl':
                score = common.SCORE[sex][joint_name][grade]
                return score
            elif sex == 'boy':
                score = common.SCORE[sex][joint_name][grade]
                return score
        elif joint_name == 'PIPFifth' or joint_name == 'PIPThird':
            old_grade = self.model(part_image)
            grade = torch.argmax(old_grade[0])
            if sex == 'girl':
                score = common.SCORE[sex][joint_name][grade]
                return score
            elif sex == 'boy':
                score = common.SCORE[sex][joint_name][grade]
                return score
        elif joint_name == 'PIPFirst':
            old_grade = self.model(part_image)
            grade = torch.argmax(old_grade[0])
            if sex == 'girl':
                score = common.SCORE[sex][joint_name][grade]
                return score
            elif sex == 'boy':
                score = common.SCORE[sex][joint_name][grade]
                return score
        elif joint_name == 'Radius':
            old_grade = self.model(part_image)
            grade = torch.argmax(old_grade[0])
            if sex == 'girl':
                score = common.SCORE[sex][joint_name][grade]
                return score
            elif sex == 'boy':
                score = common.SCORE[sex][joint_name][grade]
                return score
        elif joint_name == 'Ulna':
            old_grade = self.model(part_image)
            grade = torch.argmax(old_grade[0])
            if sex == 'girl':
                score = common.SCORE[sex][joint_name][grade]
                return score
            elif sex == 'boy':
                score = common.SCORE[sex][joint_name][grade]
                return score


    def run(self, part_image, joint_name, sex, choose_mode):
        best_acc = 0
        if choose_mode == 'train':
            for epochs in range(100):
                self.train_open(epochs)
        # elif choose_mode == 'val':
        #     for epochs in range(100):
                self.test(epochs, best_acc)
        elif choose_mode == 'detect':
            part_score = self.detect(part_image, joint_name, sex)
            return part_score


if __name__ == '__main__':
    image_path = './MCPFirst_43483.png'
    part_image = Image.open(image_path)
    gray_image = part_image.convert('L')
    to_tensor = ToTensor()
    part_image = to_tensor(gray_image)
    classifier = classifier('DIPFirst', 'train')
    run = classifier.run(part_image, 'MCPFirst', 'boy', 'detect')

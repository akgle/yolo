import torch
from BoneAge_tool import common
from PIL import ImageDraw, Image
from main_train import classifier
from torchvision.transforms import ToTensor

class BoneAge():
    def __init__(self):
        self.Radius = []
        self.Ulna = []
        self.MCPFirst = []
        self.ProximalPhalanx = []
        self.DistalPhalanx = []
        self.MiddlePhalanx = []
        self.MCP = []
        self.arthrosis_dic = {}
        self.score_list = []


    def yolov5_model(self):
        self.model = torch.hub.load(repo_or_dir=r'D:\pythonprofessor\Yolov5\yolov5-7.0', model='custom',
                               path=r'D:\pythonprofessor\Yolov5\yolov5-7.0\run\train\exp21\weights\best.pt',
                               source='local')
        self.model.eval()
        self.model.conf = 0.6
        return self.model

    def detect(self, image_path, sex, mode):
        # 初始化网络
        v5_model = self.yolov5_model()
        # 获取相片矩形框的信息
        information = v5_model(image_path)
        # 确认性别
        self.is_sex(sex)
        # 判断是否有21个关节点
        if self.is_21joint(information.xyxy[0].shape[0]):
            # 这里的information的shape为(21,6),外加列表，这里循环去除列表得到每个框的信息
            for idx, text in enumerate(information.xyxy[0][:, 5]):
                if text == 0:
                    self.Radius.append(information.xyxy[0][idx].tolist())
                elif text == 1:
                    self.Ulna.append(information.xyxy[0][idx].tolist())
                elif text == 2:
                    self.MCPFirst.append(information.xyxy[0][idx].tolist())
                elif text == 3:
                    self.ProximalPhalanx.append(information.xyxy[0][idx].tolist())
                elif text == 4:
                    self.DistalPhalanx.append(information.xyxy[0][idx].tolist())
                elif text == 5:
                    self.MiddlePhalanx.append(information.xyxy[0][idx].tolist())
                elif text == 6:
                    self.MCP.append(information.xyxy[0][idx].tolist())
            # 取出想要的13个手指关节点
            idx_3num = [0, 2, 4]
            idx_2num = [0, 2]
            DIP = [sorted(self.DistalPhalanx)[i] for i in idx_3num]
            MIP = [sorted(self.MiddlePhalanx)[i] for i in idx_2num]
            PIP = [sorted(self.ProximalPhalanx)[i] for i in idx_3num]
            MCP = [sorted(self.MCP)[i] for i in idx_2num]
            MCPF = self.MCPFirst
            ULNA = self.Ulna
            RADIUS = self.Radius
            # 13个手指信息以列表形式拼接在一起
            self.arthrosis_info = DIP + MIP + PIP + MCP + MCPF + ULNA + RADIUS
            if self.is_left_hand():
                for i in range(len(common.arthrosis_order)):
                    self.arthrosis_dic[common.arthrosis_order[i]] = self.arthrosis_info[i]
                # 切割13块关节点,获取13张关节点图片
                cut_joint_images = self.cut_13joint(image_path, self.arthrosis_dic)
                # 画框
                self.draw_retangcle(image_path)
                # 将13个矩形框以及参数分别送入分类模型进行训练
                for cut_joint_image in cut_joint_images.items():
                    classify = classifier(cut_joint_image[0], mode)
                    # 期待返回值是该关节预测分类数
                    part_score = classify.run(cut_joint_image[1], cut_joint_image[0], sex, mode)
                    self.score_list.append(part_score)
                # 总分计算
                total_score = sum(self.score_list)
                # 骨龄计算
                bone_age = common.calcBoneAge(total_score, sex)
                print(f"骨龄为：{bone_age}")
                # 输出参数
                common.export()
            else:
                print('请正确放置左手')
                exit()
        else:
            print('检测错误，关节点不符合要求')
            exit()


    def draw_retangcle(self, image_path):
        # 输入13个关节点的信息：名字+坐标信息
        # 没转成pil格式后面的pil_image.show()用不出来
        image = Image.open(image_path)
        # 框与名字显示不出来颜色，将其转化为RGB图像
        image = image.convert("RGB")
        for content in self.arthrosis_dic.items():
            lc = content[1][:4]
            name = content[0]
            image_show = ImageDraw.Draw(image)
            image_show.rectangle(xy=lc, width=2, outline='red')
            image_show.text(xy=lc[0:2], text=name, fill='red')
        # image.show()


    def cut_13joint(self, image_path, joint_content):
        # 根据传过来的标签和坐标信息，切割13个关节点
        images = {}
        image = Image.open(image_path)
        # 将图片转为灰度图
        gray_image = image.convert('L')
        for joint in joint_content.items():
            x1, y1, x2, y2 = joint[1][0:4]
            joint_image = gray_image.crop((x1, y1, x2, y2))
            joint_image.show()
            to_tensor = ToTensor()
            joint_image = to_tensor(joint_image)
            images[joint[0]] = joint_image
        return images


    def is_left_hand(self):
        if self.arthrosis_info[11][0] < self.arthrosis_info[12][0]:
            print('放置的是：左手')
            return True
        else:
            return False

    def is_21joint(self, content):
        if content == 21:
            print('存在21个关节点')
            return True
        else:
            print(f'信息错误，存在{content}个关节点')
            return False

    def is_sex(self, sex):
        if sex == 'boy':
            print('性别：男')
        elif sex == 'girl':
            print('性别：女')
        else:
            print('性别：其他')


if __name__ == '__main__':
    sex = input('请输入性别：boy or girl or other')
    image_path = r'D:\pythonprofessor\Yolov5\1526.png'
    imfo = BoneAge()
    new_image = imfo.detect(image_path, sex, mode='detect')


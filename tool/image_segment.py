from ultralytics import YOLO
import cv2
import numpy as np
import os
import tqdm

def cut_img(image, points, box):
    #创建一个与原图大小相同的掩码图，初始为黑色
    mask = np.zeros_like(image)
    #将轮廓点坐标转换为整数类型
    points_int = np.int32(points)
    #在掩码上绘制白色多边形
    cv2.fillConvexPoly(mask,points_int,(255,255,255))
    #使用掩码从原图中提取出目标区域
    img_cut = cv2.bitwise_and(image,mask)
    x, y, w, h = cv2.boundingRect(box)
    cropped_img = img_cut[y:y + h, x:x + w]
    return cropped_img

if __name__ == '__main__':
    j =0
    read_path = r"D:\pythonprofessor\yolo\dataset\images\val"
    write_path = r"D:\pythonprofessor\yolo\zhong_dataset\images\val"
    file_list = os.listdir(read_path)
    for idx, filename in enumerate(file_list):
        if filename.endswith(('.png', '.jpg', 'jpeg', 'bmp')):
            image_path = os.path.join(read_path, filename)
            img = cv2.imread(image_path)
            model = YOLO(r"D:\pythonprofessor\yolo\train2\weights\best.pt")
            out = model.predict(task="segment", source=img, show=False, save=False, conf=0.9, show_labels=False,
                                show_conf=False,
                                show_boxes=False)
            if out[0] is not None and out[0].masks is not None:
                number = len(out[0].masks.xy)
                names = out[0].names
                cls = (out[0].boxes.cls).cpu().numpy()
                len_cls = len((out[0].boxes.cls).cpu().numpy())
                # conf = np.array(out[0].boxes.conf)
                i = 0
                for i in tqdm.tqdm(range(number), desc=f'第{idx+1}张图'):
                    points = np.array(out[0].masks.xy[i], dtype=np.int32)
                    ls = points.reshape(-1, 1, 2)
                    rect = cv2.minAreaRect(ls)
                    cd = cv2.boxPoints(rect)
                    box = np.int32(cd).reshape(-1, 4, 2)
                    # img_contour = cv2.polylines(img, box, True, (0, 0, 255), 3)
                    out_img = cut_img(img, points, box)
                    name = names[cls[i]]
                    if not os.path.exists(os.path.join(write_path, f'{name}')):
                        os.makedirs(os.path.join(write_path, f'{name}'))
                    save_path = os.path.join(write_path, f'{name}/{j}.png')
                    # save_path = os.path.join(write_path, f'val_image{idx}.jpg')
                    cv2.imwrite(save_path, out_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    j=j+1
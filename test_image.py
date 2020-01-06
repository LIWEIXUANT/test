import cv2
import numpy as np
# mean = np.array([127.5, 127.5, 127.5])
# std = np.array([127.5, 127.5, 127.5])
# img = cv2.imread("5.jpg")
# img = (img - mean) / std
# print(img)
bbox = np.asarray([[20, 30, 400, 500], [300, 400, 500, 600]], dtype=np.float32) # [y1, x1, y2, x2] format
sub_sample = 16
feature_size = (800 // sub_sample)
ctr_x = np.arange(sub_sample, (feature_size + 1) * sub_sample, sub_sample)  # 共feature_size个
ctr_y = np.arange(sub_sample, (feature_size + 1) * sub_sample, sub_sample)  # 共feature_size个
print(ctr_x)
# print(ctr_y)
index = 0
# ctr: 每个网格的中心点，一共feature_size*feature_size个网格
ctr = dict()
for x in range(len(ctr_x)):
    for y in range(len(ctr_y)):
        ctr[index] = [-1, -1]
        ctr[index][1] = ctr_x[x] - 8  # 右下角坐标 - 8 = 中心坐标
        ctr[index][0] = ctr_y[y] - 8
        index += 1
#print(ctr)
ratios = [0.5, 1, 2]
anchor_scales = [8, 16, 32]  # 该尺寸是针对特征图的
anchors = np.zeros(((feature_size * feature_size * 9), 4))  # (22500, 4)
index = 0
# # 将候选框的坐标赋值到anchors
for c in ctr:
    ctr_y, ctr_x = ctr[c]
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            # anchor_scales 是针对特征图的，所以需要乘以下采样"sub_sample"
            h = sub_sample * anchor_scales[j] * np.sqrt(ratios[i])

            w = sub_sample * anchor_scales[j] * np.sqrt(1. / ratios[i])
            anchors[index, 0] = ctr_y - h / 2.
            anchors[index, 1] = ctr_x - w / 2.
            anchors[index, 2] = ctr_y + h / 2.
            anchors[index, 3] = ctr_x + w / 2.
            index += 1


valid_anchor_index = np.where(
        (anchors[:, 0] >= 0) &
        (anchors[:, 1] >= 0) &
        (anchors[:, 2] <= 800) &
        (anchors[:, 3] <= 800)
    )[0]  # 该函数返回数组中满足条件的index
print (valid_anchor_index.shape) # (8940,)，表明有8940个框满足条件

 # 获取有效anchor（即边框都在图片内的anchor）的坐标
valid_anchor_boxes = anchors[valid_anchor_index]

# 计算有效anchor框"valid_anchor_boxes"与目标框"bbox"的IOU

print(valid_anchor_boxes[0])
valid_anchor_num = len(valid_anchor_boxes)
ious = np.empty((valid_anchor_num, 2), dtype=np.float32)
ious.fill(0)
print(ious.shape)
for num1, i in enumerate(valid_anchor_boxes):
    ya1, xa1, ya2, xa2 = i
    anchor_area = (ya2 - ya1) * (xa2 - xa1)  # anchor框面积
    print(ya1, xa1, ya2, xa2)

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import os
from PIL import Image

class Histobags(data_utils.Dataset):
# 初始化函数，设置数据集的路径、转换、标签等属性
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir # 数据集的根目录
        self.transform = transform 
        self.labels = np.array([0, 1, 2, 3, 4]) # 数据集的标签列表 0=Pos, 1=AGC, 2=AGC-FN, 3=AIS, 4=ADC

  # 返回数据集的长度，即样本（实例集合）的数量
    def __len__(self):
        return len(os.listdir(self.root_dir)) 

  # 根据给定的索引返回一个样本（实例集合）及其标签
    def __getitem__(self, idx):
        bag_name = os.listdir(self.root_dir)[idx] # 根据索引获取样本文件夹名
        bag_path = os.path.join(self.root_dir, bag_name) # 拼接样本文件夹路径
        instances = [] # 创建一个空列表来存储实例张量

        for instance_name in os.listdir(bag_path): # 遍历样本文件夹下的所有实例文件名
            instance_path = os.path.join(bag_path, instance_name) # 拼接实例文件路径
            image = Image.open(instance_path) 

            if self.transform: # 如果有转换函数，就对图片进行转换
                image = self.transform(image)

            instances.append(image) # 将图片张量添加到实例列表中
        #print(bag_name)
        instances = torch.stack(instances) # 将实例列表转换为一个张量，形状为(实例数，通道数，高度，宽度)
        if "Pos" in bag_name:
            label = self.labels[0]
        elif "AGC-FN" in bag_name:
            label = self.labels[2]
        elif "AIS" in bag_name:
            label = self.labels[3]
        elif "ADC" in bag_name:
            label = self.labels[4]
        elif "AGC" in bag_name:
            label = self.labels[1]
        #label = torch.from_numpy(label)
        return instances, label # 返回样本（实例集合）和标签
    

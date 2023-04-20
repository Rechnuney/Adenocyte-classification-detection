import os
import numpy as np
import random
import shutil
import socket

split_names = ['train', 'test']
cls_names = ['Pos', 'AGC', 'AGC-FN', 'AIS', 'ADC']
dst_dir = '/remote-home/share/DATA/RedHouse/AdenocyteBag/abmil_0326data_512_180'

for split_name in split_names:
    split_path = os.path.join(dst_dir, split_name)
    if os.path.exists(split_path) == False:
        os.mkdir(split_name)

    # for cls_name in cls_names:
    #     cls_path = os.path.join(split_path, cls_name)
    #     if os.path.exists(cls_path) == False:
    #         os.mkdir(cls_path)

data_path = '/remote-home/share/DATA/RedHouse/AdenocyteBag/abmil_0326data_512_180/bags_cut_64'

bag_names = os.listdir(data_path)
bag_labels = []
Pos_bag = []
AGC_bag = []
AGC_FN_bag = []
AIS_bag = []
ADC_bag = []

for bag_name in bag_names:
    if cls_names[0] in bag_name:
        bag_labels.append(0)
        Pos_bag.append(bag_name)
    elif cls_names[2] in bag_name:
        bag_labels.append(2)
        AGC_FN_bag.append(bag_name)
    elif cls_names[3] in bag_name:
        bag_labels.append(3)
        AIS_bag.append(bag_name)
    elif cls_names[4] in bag_name:
        bag_labels.append(4)
        ADC_bag.append(bag_name)
    elif cls_names[1] in bag_name:
        bag_labels.append(1)
        AGC_bag.append(bag_name)

Pos_num = len(Pos_bag)
AGC_num = len(AGC_bag) 
AGC_FN_num = len(AGC_FN_bag)
AIS_num = len(AIS_bag)
ADC_num = len(ADC_bag)
print(Pos_num, AGC_num, AGC_FN_num, AIS_num, ADC_num)

train_Pos_num = int(Pos_num*0.8)
train_AGC_num = int(AGC_num*0.8)
train_AGC_FN_num = int(AGC_FN_num*0.8)
train_AIS_num = int(AIS_num*0.8)
train_ADC_num = int(ADC_num*0.8)

random.seed(1)
random.shuffle(Pos_bag)

random.seed(1)
random.shuffle(AGC_bag)

random.seed(1)
random.shuffle(AGC_FN_bag)

random.seed(1)
random.shuffle(AIS_bag)

random.seed(1)
random.shuffle(ADC_bag)


train_Pos_bag = Pos_bag[:train_Pos_num]
train_AGC_bag = AGC_bag[:train_AGC_num]
train_AGC_FN_bag = AGC_FN_bag[:train_AGC_FN_num]
train_AIS_bag = AIS_bag[:train_AIS_num]
train_ADC_bag = ADC_bag[:train_ADC_num]
train_bag = train_Pos_bag + train_AGC_bag + train_AGC_FN_bag + train_AIS_bag + train_ADC_bag

test_Pos_bag = Pos_bag[train_Pos_num:]
test_AGC_bag = AGC_bag[train_AGC_num:]
test_AGC_FN_bag = AGC_FN_bag[train_AGC_FN_num:]
test_AIS_bag = AIS_bag[train_AIS_num:]
test_ADC_bag = ADC_bag[train_ADC_num:]
test_bag = test_Pos_bag + test_AGC_bag + test_AGC_FN_bag + test_AIS_bag + test_ADC_bag


for i in train_bag:
    shutil.copytree(os.path.join(data_path, i), os.path.join(dst_dir, 'train', i))
    print("one",i)
for i in test_bag:
    shutil.copytree(os.path.join(data_path, i), os.path.join(dst_dir, 'test', i))
    print("two",i)

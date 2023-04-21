import os
import numpy as np
import random
import shutil
import sys
#import socket

split_names = ['train', 'test', 'val']
cls_names = ['positive', 'negative']
dst_dir = '/remote-home/share/DATA/RedHouse/AdenocyteBag/abmil_0118data_768_448'


data_path = '/remote-home/share/DATA/RedHouse/AdenocyteBag/abmil_0118data_768_448/bags_cut_64'
print(data_path)

bags = os.listdir(data_path)
bag_labels = []
pos_bag = []
neg_bag = []

for bag in bags:
    if 'positive' in bag:
        bag_labels.append(1)
        pos_bag.append(bag)
    elif 'negative' in bag:
        bag_labels.append(0)
        neg_bag.append(bag)

pos_num = len(pos_bag)
neg_num = len(neg_bag) 
print(pos_num, neg_num)

train_pos_num = int(pos_num*0.6)
train_neg_num = int(neg_num*0.6)

test_pos_num = int(pos_num*0.2)
test_neg_num = int(neg_num*0.2)

random.seed(1)
random.shuffle(pos_bag)
random.seed(1)
random.shuffle(neg_bag)

train_pos_bag = pos_bag[:train_pos_num]
train_neg_bag = neg_bag[:train_neg_num]
train_bag = train_pos_bag + train_neg_bag

test_pos_bag = pos_bag[train_pos_num: train_pos_num+test_pos_num]
test_neg_bag = neg_bag[train_neg_num: train_neg_num+test_neg_num]
test_bag = test_pos_bag + test_neg_bag

val_pos_bag = pos_bag[train_pos_num+test_pos_num:]
val_neg_bag = neg_bag[train_neg_num+test_neg_num:]
val_bag = val_pos_bag + val_neg_bag

i = 0

print(len(train_bag), len(test_bag), len(val_bag))

for bag in train_bag:
    i += 1
    print(os.path.join(data_path, bag))
    sys.exit(0)
    shutil.copytree(os.path.join(data_path, bag), os.path.join(dst_dir, 'train', bag))
    print(i)
for bag in test_bag:
    i += 1
    shutil.copytree(os.path.join(data_path, bag), os.path.join(dst_dir, 'test', bag))
    print(i)
for bag in val_bag:
    i += 1
    shutil.copytree(os.path.join(data_path, bag), os.path.join(dst_dir, 'val', bag))
    print(i)


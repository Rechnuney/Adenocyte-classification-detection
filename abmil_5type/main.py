from __future__ import print_function

import numpy as np
import time
import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from dataloader import Histobags
from model import Attention, GatedAttention
from sklearn import metrics 
import matplotlib as plt
# Training settings
perf_path= '/remote-home/chenyuren/abmil_5type/performance2.txt'

parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--target_number', type=int, default=9, metavar='T',
                    help='bags have a positive labels if they contain at least one 9')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

#加载数据集

print('Load Train and Test Set')

loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

transform = transforms.Compose([

    # 1、centercrop图像中心裁剪
    # 从图像的中心位置进行裁剪
#     transforms.CenterCrop(122),
    # 裁剪的尺寸比图像原始尺寸大，会怎么做？
    # 会在图像的周围进行0填充，也就是黑色，1代表白色
#     transforms.CenterCrop(512),
 
    #2、RandomCrop随机裁剪操作
#     transforms.RandomCrop(224,padding=16),
#     transforms.RandomCrop(224,padding=(16,64)),
    # 指定某个颜色进行填充
#     transforms.RandomCrop(224,padding=16,fill=(255,0,0)),
#     transforms.RandomCrop(512,pad_if_needed=True),
#     transforms.RandomCrop(224,padding=64,padding_mode='edge'),
#     transforms.RandomCrop(224,padding=64,padding_mode='reflect'),
#     transforms.RandomCrop(1024,padding=1024,padding_mode='symmetric'),
    
    # 3、RandomResizedCrop
    # 随机scale的范围是（0.5，0.5）这个可以自己进行调整
#     transforms.RandomResizedCrop(size=224,scale=(0.5,0.5)),
    
    # 4、FiveCrop:返回的是一个四维的结果
#     transforms.FiveCrop(112),
#     transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops])),
    
    # 5、TenCrop
    # transforms.TenCrop(112, vertical_flip=False),
    # transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops])),
    
    # 6、Horizontal Flip 水平翻转
#    transforms.RandomHorizontalFlip(),
    
    # 7、vertical Flip 垂直翻转
#     transforms.RandomVerticalFlip(p=0.5),
    
    # 8、RandomRotation 随机旋转指定的角度
#    transforms.RandomRotation(90),
#     transforms.RandomRotation((90), expand=True),
#     transforms.RandomRotation(30, center=(0, 0)),
#     transforms.RandomRotation(30, center=(0, 0), expand=True),   # expand only for center rotation
    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7336, 0.7470, 0.8317], std=[0.1515, 0.1421, 0.0947])
])

# 创建一个数据集对象，指定根目录和转换函数（这里只是随机裁剪和转换为张量）
train_dataset = Histobags(root_dir="/remote-home/share/DATA/RedHouse/AdenocyteBag/abmil_0326data_512_180/train", transform=transform)
test_dataset = Histobags(root_dir="/remote-home/share/DATA/RedHouse/AdenocyteBag/abmil_0326data_512_180/test", transform=transform)
val_dataset = Histobags(root_dir="/remote-home/share/DATA/RedHouse/AdenocyteBag/abmil_0326data_512_180/val", transform=transform)
# 创建一个数据加载器对象，指定批次大小为1，打乱顺序为True
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)    
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)
print("dataloader created")

# 迭代数据加载器对象，获取每个批次的样本（实例集合）和标签，并打印它们的形状（这里只迭代一次）
#for bags, labels in train_loader:
#    print(bags.shape) # 应该是(1, 实例数, 3, 32, 32)，即(批次大小，实例数，通道数，高度，宽度)
#    print(type(labels)) 
#    break


print('Init Model')
if args.model=='attention':
    model = Attention()
elif args.model=='gated_attention':
    model = GatedAttention()
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

weight = torch.tensor([1765/148, 1765/437, 1765/234, 1765/357, 1]).cuda()

def train(epoch):
    time_start = time.time()
    model.train()
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, label) in enumerate(train_loader):
        bag_label = label
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics

        loss, _ = model.calculate_objective(data, bag_label, weight)
        train_loss += loss.data

        error, _ , _= model.calculate_classification_error(data, bag_label)
        train_error += error
        # backward pass
        loss.backward()
        # step
        optimizer.step()

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    time_end = time.time()
    time_c= time_end - time_start 
    print('Epoch: {}/{}, Train Loss: {:.4f}, Train error: {:.4f}, Time_cost:{:.4f}'.format(epoch, args.epochs, train_loss.cpu().numpy(), train_error, time_c))
    f = open(perf_path, 'a+')
    f.write('Epoch: {}/{}, Train Loss: {:.4f}, Train error: {:.4f}, Time_cost:{:.4f}'.format(epoch, args.epochs, train_loss.cpu().numpy(), train_error, time_c))
    f.write('\n')
    f.close()


def val():
    time_start = time.time()
    model.eval()
    val_loss = 0.
    val_error = 0.
    real_labels = np.array([])
    predicted_labels = np.array([])

    for batch_idx, (data, label) in enumerate(val_loader):
        bag_label = label
        real_labels = np.append(real_labels, int(label.data.numpy()[0]))
        
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        loss, attention_weights = model.calculate_objective(data, bag_label, weight)
        val_loss += loss.data
        error, predicted_label, y_score = model.calculate_classification_error(data, bag_label)
        val_error += error

        predicted_label = int(predicted_label.cpu().data.numpy()[0])
        predicted_labels = np.append(predicted_labels, predicted_label)

    # calculate loss and error for epoch
    val_loss /= len(val_loader)
    val_error /= len(val_loader)

    time_end = time.time()
    time_c= time_end - time_start 
    print('Val Loss: {:.4f}, val error: {:.4f}, time cost: {:.4f}'.format(val_loss.cpu().numpy(), val_error, time_c))
    print("confusion matrix = \n", metrics.confusion_matrix(real_labels, predicted_labels), '\n')
    
    f = open(perf_path, 'a+')
    f.write('\nVal: Loss: {:.4f}, val error: {:.4f}, time cost: {:.4f}'.format(val_loss.cpu().numpy(), val_error, time_c))
    f.write('\n')
    f.close()
    return val_error, val_loss, real_labels, predicted_labels

def test():
    model.eval()
    test_loss = 0.
    test_error = 0.
    real_labels = np.array([])
    predicted_labels = np.array([])
    y_scores = np.empty((0,5), float)

    for batch_idx, (data, label) in enumerate(test_loader):
        bag_label = label
        real_labels = np.append(real_labels, int(label.data.numpy()[0]))
        #instance_labels = label[1]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        loss, attention_weights = model.calculate_objective(data, bag_label, weight)
        test_loss += loss.data
        error, predicted_label, y_score = model.calculate_classification_error(data, bag_label)
        test_error += error

        predicted_label = int(predicted_label.cpu().data.numpy()[0])
        predicted_labels = np.append(predicted_labels, predicted_label)

        y_score = y_score.cpu().data.numpy()
        y_scores = np.append(y_scores, [y_score], axis=0)
        

        '''if batch_idx < 5:  # plot bag labels and instance labels for first 5 bags
            bag_level = (bag_label.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0][0]))
            instance_level = list(zip(instance_labels.numpy()[0].tolist(),
                                 np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()))

            print('\nTrue Bag Label, Predicted Bag Label: {}\n'
                  'True Instance Labels, Attention Weights: {}'.format(bag_level, instance_level))'''

    test_error /= len(test_loader)
    test_loss /= len(test_loader)

    print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss.cpu().numpy(), test_error))
    return real_labels, predicted_labels, y_scores

if __name__ == "__main__":
    target_names = ['Pos', 'AGC', 'AGC_FN', 'AIS', 'ADC']
    lowest_error = 1
    best_epoch = 0

    f = open(perf_path, 'a+')
    f.write('lr: {},  epoch: {:.4f}'.format(args.lr, args.epochs))
    f.write('\n')
    f.close()

    print('Start Training')
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        if epoch % 5 == 0:
            val_error, val_loss, real_labels, predicted_labels = val()

            if val_error < lowest_error:
                lowest_error = val_error
                best_epoch = epoch 

            print('Val:\t Lowest error: {:.4f}, Best epoch: {}\n'.format(lowest_error, best_epoch))
            print("confusion matrix = \n", metrics.confusion_matrix(real_labels, predicted_labels))
            f = open(perf_path, 'a+')
            f.write("confusion matrix = \n")
            f.write(str(metrics.confusion_matrix(real_labels, predicted_labels)))
            f.write('\n')   
            f.write('Val:\t Lowest error: {:.4f}, Best epoch: {}\n'.format(lowest_error, best_epoch))
            f.write('\n')
            f.close()

    print('Start Testing')

    real_labels, predicted_labels, y_scores = test()

    print("confusion matrix = \n", metrics.confusion_matrix(real_labels, predicted_labels))
    print(metrics.classification_report(real_labels, predicted_labels, target_names=target_names))


    # auc_macro = metrics.roc_auc_score(real_labels, y_scores, average='macro', multi_class='ovo')
    # auc_micro = metrics.roc_auc_score(real_labels, y_scores, average='micro', multi_class='ovo')

    # print("roc_macro = ", roc_macro)
    # print("roc_micro = ", roc_micro)
    
    fpr, tpr, thresholds = metrics.roc_curve(np.eye(5)[real_labels], y_scores) #将real_labels变成ine-hot编码
    roc_auc = metrics.auc(fpr, tpr)
    print(roc_auc)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                     estimator_name='example estimator')
    display.plot()

    plt.show()
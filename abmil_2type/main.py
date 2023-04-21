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
weight = torch.tensor([1., 2.]).cuda()

perf_path = '/remote-home/chenyuren/abmil_2type/performance2.txt'

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
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
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7336, 0.7470, 0.8317], std=[0.1515, 0.1421, 0.0947])
])

# 创建一个数据集对象，指定根目录和转换函数（这里只是随机裁剪和转换为张量）
train_dataset = Histobags(root_dir="/remote-home/share/DATA/RedHouse/AdenocyteBag/abmil_0118data_768_448/train", transform=transform)
test_dataset = Histobags(root_dir="/remote-home/share/DATA/RedHouse/AdenocyteBag/abmil_0118data_768_448/test", transform=transform)
val_dataset = Histobags(root_dir="/remote-home/share/DATA/RedHouse/AdenocyteBag/abmil_0118data_768_448/val", transform=transform)
#train_dataset = Histobags(root_dir="/remote-home/share/DATA/RedHouse/AdenocyteBag/abmil_0118data_512_128/abmil_try/train", transform=transform)
#test_dataset = Histobags(root_dir="/remote-home/share/DATA/RedHouse/AdenocyteBag/abmil_0118data_512_128/abmil_try/test", transform=transform)
#val_dataset = Histobags(root_dir="/remote-home/share/DATA/RedHouse/AdenocyteBag/abmil_0118data_512_128/abmil_try/test", transform=transform)

# 创建一个数据加载器对象，指定批次大小为1，打乱顺序为True
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)    
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)
print("dataloader created")


print('Init Model')
if args.model=='attention':
    model = Attention()
elif args.model=='gated_attention':
    model = GatedAttention()
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)



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
    print('\nVal: Loss: {:.4f}, val error: {:.4f}, time cost: {:.4f}'.format(val_loss.cpu().numpy(), val_error, time_c))
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
    y_scores = np.empty((0,2), float)

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
    f = open(perf_path, 'a+')
    f.write('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss.cpu().numpy(), test_error))
    f.write('\n')
    f.close()

    return real_labels, predicted_labels, y_scores

if __name__ == "__main__":
    target_names = ['negative', 'positive']
    f = open(perf_path, 'a+')
    f.write('lr: {},  epoch: {:.4f}, weight:{}'.format(args.lr, args.epochs, weight))
    f.write('\n')
    f.close()

    lowest_error = 1
    best_epoch = 0

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


    f = open(perf_path, 'a+')
    f.write('Validation set:\t Best accuracy: {:.4f}, Best epoch: {}\n'.format(lowest_error, best_epoch))
    f.write("confusion matrix = \n")
    f.write(str(metrics.confusion_matrix(real_labels, predicted_labels)))
    f.write(metrics.classification_report(real_labels, predicted_labels, target_names=target_names))
    f.write('\n\n')
    f.close()

    fpr, tpr, thresholds = metrics.roc_curve(real_labels, y_scores)
    roc_auc = metrics.auc(fpr, tpr)
    print('roc_auc = ', roc_auc)

    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                     estimator_name='example estimator')
    display.plot()

    plt.show()
    
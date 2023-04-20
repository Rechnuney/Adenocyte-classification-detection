#encoding: utf-8
import torch
import os
import sys
import numpy as np
from my_dataloader import MyDataset
from model import ResNet_s, BasicBlock
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import argparse
from sklearn import metrics 

parser = argparse.ArgumentParser(description='PyTorch resnet Example')
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

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

perf_path = '/remote-home/chenyuren/resnet_5type/performance.txt'

def train(train_dl, model, criteria, optimizer, epoch):
    print('Epoch {}/{}'.format(epoch+1, epochs_num))
    f = open(perf_path, 'a+')
    f.write('Epoch {}/{}'.format(epoch+1, epochs_num))
    f.write('\n')
    f.close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.train()
    train_loss = 0
    train_correct = 0

    for data, target in train_dl:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        
            
        loss = criteria(output, target.long())
        train_loss += loss.item() * data.size(0)
            
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)

        train_correct += pred.eq(target.long().view_as(pred)).sum().item()
        
    epoch_loss = train_loss / len(train_dl.dataset)
    epoch_acc = train_correct / len(train_dl.dataset)
    
    print('Train set:  Loss: {:.4f}, Accuracy: {:.2f}%'.format(
        epoch_loss,  100.0 * epoch_acc
        ))
    f = open(perf_path, 'a+')
    f.write('Train set:  Loss: {:.4f}, Accuracy: {:.2f}%'.format(
        epoch_loss,  100.0 * epoch_acc
        ))
    f.write('\n')
    f.close()


def validate(val_dl, model, criteria):
    model.eval()
    loss = 0
    correct = 0
    real_labels = np.array([])
    predicted_labels = np.array([])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for data, target in val_dl:
            data, target = data.to(device), target.to(device)

            real_labels = np.append(real_labels, target.cpu().data.numpy())
            output = model(data)

            loss += criteria(output, target.long()).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)

            predicted_labels = np.append(predicted_labels, pred.cpu().data.numpy())

            correct += pred.eq(target.long().view_as(pred)).sum().item()
    
    val_loss = loss / len(val_dl.dataset)
    val_acc = correct / len(val_dl.dataset)

    print('Validation set:   Loss: {:.4f}, Accuracy: {:.2f}%'.format(
        val_loss,  100.0 * val_acc
        ))
    f = open(perf_path, 'a+')
    f.write('Validation set:   Loss: {:.4f}, Accuracy: {:.2f}%'.format(
        val_loss,  100.0 * val_acc
        ))
    f.write('\n')
    f.close()

    return val_loss, val_acc, real_labels, predicted_labels
    
def test(test_dl, model, criteria):
    model.eval()
    loss = 0
    correct = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real_labels = np.array([])
    predicted_labels = np.array([])

    with torch.no_grad():
        for data, target in test_dl:
            data, target = data.to(device), target.to(device)

            real_labels = np.append(real_labels, target.cpu().data.numpy())
            output = model(data)

            loss += criteria(output, target.long()).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            predicted_labels = np.append(predicted_labels, pred.cpu().data.numpy())

            correct += pred.eq(target.long().view_as(pred)).sum().item()
    
    test_loss = loss / len(test_dl.dataset)
    test_acc = correct / len(test_dl.dataset)

    print('test set:   Loss: {:.4f}, Accuracy: {:.2f}%'.format(
        test_loss, 100.0 * test_acc
        ))
    f = open(perf_path, 'a+')
    f.write('test set:   Loss: {:.4f}, Accuracy: {:.2f}%'.format(
        test_loss, 100.0 * test_acc
        ))
    f.write('\n')
    f.close()

    return real_labels, predicted_labels


if __name__ == '__main__':
    batchsize = 4
    epochs_num = args.epochs
    target_names = ['Pos', 'AGC', 'AGC_FN', 'AIS', 'ADC']
    weight = torch.tensor([1765/148, 1765/437, 1765/234, 1765/357, 1]).cuda()

    f = open(perf_path, 'a+')
    f.write('lr: {},  epoch: {:.4f}'.format(args.lr, epochs_num))
    f.write('\n')
    f.close()

    if torch.cuda.is_available():
        loader_kwargs = {'num_workers': 1, 'pin_memory': True} 

    # 准备数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainset = MyDataset(phase='train', data_path='/remote-home/share/DATA/RedHouse/AdenocyteBag/abmil_0326data_512_180/resnet', transform=transform)
    testset = MyDataset(phase='test', data_path='/remote-home/share/DATA/RedHouse/AdenocyteBag/abmil_0326data_512_180/resnet', transform=transform)
    valset = MyDataset(phase='val', data_path='/remote-home/share/DATA/RedHouse/AdenocyteBag/abmil_0326data_512_180/resnet', transform=transform)

    print(trainset.__len__())
    print(testset.__len__())
    print(valset.__len__())

    train_dl = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True, **loader_kwargs)
    test_dl = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False, **loader_kwargs)
    val_dl = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=False, **loader_kwargs)

    model = ResNet_s(BasicBlock, [2,2,2,2], num_classes=5)
    if torch.cuda.is_available():
        model = model.cuda()
        torch.cuda.empty_cache()

    # 训练模型
    criteria = nn.CrossEntropyLoss(weight=weight)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=10e-5)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    best_acc = 0
    best_epoch = 0

    for epoch in range(epochs_num):
        train(train_dl, model, criteria, optimizer, epoch)
       
        _, val_acc, real_labels, predicted_labels = validate(val_dl, model, criteria)
        print("confusion matrix = \n", metrics.confusion_matrix(real_labels, predicted_labels))
        f = open(perf_path, 'a+')
        f.write("confusion matrix = \n")
        f.write(str(metrics.confusion_matrix(real_labels, predicted_labels)))
        f.write('\n')
        f.close()
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
        
    print('Validation set:\t Best accuracy: {:.4f}, Best epoch: {}\n'.format(best_acc, best_epoch))

    real_labels, predicted_labels = test(test_dl, model, criteria)
    print("confusion matrix = \n", metrics.confusion_matrix(real_labels, predicted_labels))
    print(metrics.classification_report(real_labels, predicted_labels, target_names=target_names))

    f = open(perf_path, 'a+')
    f.write('Validation set:\t Best accuracy: {:.4f}, Best epoch: {}\n'.format(best_acc, best_epoch))
    f.write("confusion matrix = \n")
    f.write(str(metrics.confusion_matrix(real_labels, predicted_labels)))
    f.write(metrics.classification_report(real_labels, predicted_labels, target_names=target_names))
    f.write('\n\n')
    f.close()
    
  


           
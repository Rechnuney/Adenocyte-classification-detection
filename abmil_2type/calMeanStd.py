#时间复杂度太高，需要优化
#每个bag长度不同，采用加权平均的方法，计算均值和方差
import torch
from torchvision import datasets, transforms
import dataloader
from dataloader import Histobags

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = Histobags(root_dir="//remote-home/share/DATA/RedHouse/AdenocyteBag/abmil_0118data_512_128/train", 
                            transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)    

mean = 0.
std = 0.
tmp = torch.zeros([1, 0, 3, 64, 64])

for images, _ in train_loader:
    #print(images.shape)
    batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
    #images = images.view(batch_samples, images.size(1), -1)
    #mean += images.mean(2).sum(0)
    #std += images.std(2).sum(0)
    tmp = torch.cat((images, tmp), dim=1)

mean = torch.mean(tmp, dim=(0, 3, 4), keepdim=False)
std = torch.std(tmp, dim=(0, 3, 4), keepdim=False)
print(mean.shape, std.shape)

mean = torch.mean(mean, dim = 0)
std = torch.mean(std, dim = 0)

print(mean)
print(std)
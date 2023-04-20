import torch
import torch.utils.data as data
import os
import torchvision
from PIL import Image

class MyDataset(data.Dataset):
    def __init__(self, phase, data_path, transform=None):
        super(MyDataset, self).__init__()
        valid_phase = ['train', 'test', 'val']
        assert phase in valid_phase
        self.phase = phase
        
        if transform is None:
            self.transform = torchvision.transforms.ToTensor()
        else:
            self.transform = transform

        if phase == 'train':
            self.data_path = os.path.join(data_path, 'train_2')
        elif phase == 'test':
            self.data_path = os.path.join(data_path, 'test_2')
        elif phase == 'val':
            self.data_path = os.path.join(data_path, 'val')

        self.img_name_list = os.listdir(self.data_path)
    
    def __len__(self):
        return len(self.img_name_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.img_name_list[idx]

        if "positve" in img_name:
            label = 1
        elif "negtive" in img_name:
            label = 0

        img_path = os.path.join(self.data_path, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
        


    

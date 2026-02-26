import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd

class WaterbirdsDataset(Dataset):
    def __init__(self, csv_file, img_root, split=2, transform=None):
        """
        Args:
            csv_file (str): metadata.csv 文件路径
            img_root (str): 图像根目录路径
            split (int): 0=训练集, 1=验证集, 2=测试集
            transform (callable, optional): 图像变换
        """
        self.data = pd.read_csv(csv_file)
        #self.data = self.data[self.data['split'] == split]  # 过滤指定数据集部分
        if split == 1 or split == 2:
            self.data = self.data[self.data['split'].isin([1, 2])]
        else:
            self.data = self.data[self.data['split'] == split]  # 合并验证集和测试集  # 过滤指定数据集部分
        self.img_root = img_root
        self.transform = transform
        self.classes = ['Land bird', 'water bird']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_root, self.data.iloc[idx]['img_filename'])
        image = Image.open(img_path).convert('RGB')
        label = int(self.data.iloc[idx]['y'])  # 0=Groundbird, 1=Waterbird
        background = int(self.data.iloc[idx]['place'])  # 0=Land, 1=Water

        if self.transform:
            image = self.transform(image)

        return image, label, background
        #return image, background, background
    
# 计算 Worst Acc
def compute_worst_acc(model, data_loader, method="tcl"):
    model.eval()
    correct = {(1, 1): 0, (1, 0): 0, (0, 0): 0, (0, 1): 0}  # 统计正确预测数量
    total = {(1, 1): 0, (1, 0): 0, (0, 0): 0, (0, 1): 0}  # 统计总数量
    
    with torch.no_grad():
        for images, labels, backgrounds in data_loader:
            images, labels = images.cuda(), labels.cuda()
            #outputs = model(images)
            if method == "su" or method == "scan":
                outputs = model(images.cuda(non_blocking=True),
                        forward_pass='return_all')['output'][0]
                preds = torch.argmax(outputs, dim=1)
            elif method == "cdcv2":
                outputs = model(images, forward_pass='return_all')['output'][0]
                preds = torch.argmax(outputs, dim=1)
            else:
                outputs = model.module.forward_c(images)
                preds = torch.argmax(outputs, dim=1)
            
            for i in range(len(labels)):
                group = (labels[i].item(), backgrounds[i].item())
                total[group] += 1
                if preds[i] == labels[i]:
                    correct[group] += 1
    
    accs = {group: (correct[group] / total[group]) if total[group] > 0 else 0 for group in total}
    worst_acc = min(accs.values())
    
    print("Group Accuracies:", accs)
    print("Worst Acc:", worst_acc)
    print("total: ", total)
    return worst_acc

""" # 示例用法
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = WaterbirdsDataset(csv_file='/mnt/data/metadata.csv', img_root='/path/to/images', split=2, transform=transform)

data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# 计算 Worst Acc
# model = ... (你的训练好的模型)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# compute_worst_acc(model, data_loader, device) """

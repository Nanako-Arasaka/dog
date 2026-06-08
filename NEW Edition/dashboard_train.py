from dashboard_model import Resnet18_dashboard,Resnet50_dashboard,Resnet34_dashboard
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

# 检查是否有可用的CUDA设备，如果有则使用第一个设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = ", device)

# 对数据集图片修改的定义，主要应用于数据集上
data_transforms = transforms.Compose([
    transforms.Resize(256),  # 将图片的短边定义为256，长短边比例不变
    transforms.CenterCrop(224),  # 裁剪中心处224*224*3的区域
    transforms.ToTensor(), # 转换为张量的形式
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定义数据集文件
train_dir = r"D:\pycharm\Creativity_match\dataset\train"
valid_dir = r"D:\pycharm\Creativity_match\dataset\valid"

# 加载数据集
train_data = datasets.ImageFolder(train_dir, transform=data_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms)
print("训练集类别：{}".format(train_data.classes))
print("验证集类别：{}".format(valid_data.classes))

# 创建数据迭代器
train_dataloader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(dataset=valid_data, batch_size=32, shuffle=True)

# 查看数据集长度
train_data_size = len(train_data)
valid_data_size = len(valid_data)
print("训练集的长度为：{}".format(train_data_size))
print("验证集的长度为：{}".format(valid_data_size))

# 模型、优化器、损失函数
#Resnet18
#model = Resnet18_dashboard(num_classes=3).to(device)
#Resnet50
#model=Resnet50_dashboard(num_classes=3).to(device)
model = Resnet34_dashboard(num_classes=3).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 总训练轮次
num_epoches = 40
best_acc = 0.0
best_loss=100.0
for epoch in range(num_epoches):
    print("----------Epoch {}/{}---------".format(epoch + 1, num_epoches))
    model.train()

    total_num_train = 0  # 重置每个epoch的训练计数器
    for data in train_dataloader:
        img, label = data
        # print(img.shape)
        # print(label.shape)
        # exit(1)
        img = img.to(device)
        label = label.to(device)
        pre_label_train = model(img)
        loss_train = loss_fn(pre_label_train, label)

        loss_train.backward()
        optimizer.step()

        total_num_train += 1
        if total_num_train % 100 == 0:
            print("数据迭代次数：{}，loss：{}".format(total_num_train, loss_train.item()))

    model.eval()
    total_loss_test = 0.0
    total_accuracy = 0.0
    total_num_test = 0  # 重置每个epoch的验证计数器
    with torch.no_grad():
        for data in valid_dataloader:
            img, label = data
            img = img.to(device)
            label = label.to(device)
            pre_label_test = model(img)
            loss_test = loss_fn(pre_label_test, label)
            total_loss_test += loss_test.item()
            _, pre_label_test = torch.max(pre_label_test.data, 1)
            accuracy = torch.sum(pre_label_test == label).item()
            total_accuracy += accuracy
            total_num_test += label.size(0)

    current_acc = total_accuracy / total_num_test
    current_loss = total_loss_test / total_num_test
    print("验证集上的 loss：{}".format(total_loss_test / total_num_test))
    print("验证集上的 acc：{}".format(current_acc))
    #保存最佳的acc
    # if current_acc > best_acc:
    #     torch.save(model.state_dict(), r"D:\pycharm\Creativity_match\workspace\model_best_5_23.pth")
    #     print("保存最佳权重！")
    #     best_acc = current_acc
    #保存最佳的loss
    if current_loss < best_loss:
        torch.save(model.state_dict(), r"D:\pycharm\Creativity_match\workspace\model_rs34_best_5_exam.pth")
        print("保存最佳权重！")
        best_loss = current_loss



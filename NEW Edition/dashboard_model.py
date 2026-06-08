import torch.nn as nn
from torchvision.models import resnet18,resnet50,resnet34
import torchvision

class Resnet18_dashboard(nn.Module):
    def __init__(self, num_classes):
        super(Resnet18_dashboard, self).__init__()

        # 创建一个 resnet18模型实例，并使用预训练的权重（在这里是 IMAGENET1K_V1）
        self.model = resnet18(pretrained=True)
        # 更新模型的最后一层，以适应特定分类任务
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),  # Dropout层，丢弃20%的神经元
            nn.Linear(512, num_classes)
        )
    # 定义前向传播方法，用于指定数据如何在模型中传递
    def forward(self, x):
        return self.model(x)




class Resnet34_dashboard(nn.Module):
    def __init__(self, num_classes):
        super(Resnet34_dashboard, self).__init__()

        self.model = resnet34(pretrained=True)
        self.model.fc = nn.Linear(in_features=512,out_features=num_classes)

    # 定义前向传播方法，用于指定数据如何在模型中传递
    def forward(self, x):
        return self.model(x)
if __name__ =="__main__":
   model = Resnet34_dashboard(3)
   print(model)

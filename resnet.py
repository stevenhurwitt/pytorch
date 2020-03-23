from PIL import Image
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import torchvision.datasets as dt
import torch
import os 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PREPROCESS = transforms.Compose([transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = [0.485,0.456,0.406],std = [0.229,0.224,0.225])])


def get_dataset(train = True):
    if train:
        trainset = dt.ImageFolder(root = "./train/",transform = PREPROCESS)
        train_loader = torch.utils.data.DataLoader(trainset,batch_size = 8,shuffle=True)
        return train_loader
    else:
        testset = dt.ImageFolder(root = "./test/",transform = PREPROCESS)
        test_loader = torch.utils.data.DataLoader(trainset,batch_size = 8,shuffle=True)
        return test_loader

class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
                        nn.BatchNorm2d(outchannel),
                        )
        self.conv2  = nn.Sequential(
                        nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(outchannel)
                    )
        self.skip = nn.Sequential()
        if stride != 1 or inchannel != self.expansion * outchannel:
            self.skip = nn.Sequential(
                nn.Conv2d(inchannel, self.expansion * outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * outchannel)
            )

    def forward(self, X):
        out = F.relu(self.conv1(X))
        out = self.conv2(out)
        out += self.skip(X)
        out = F.relu(out)
        return out

class Model(nn.Module):
    def __init__(self, ResidualBlock, num_classes):
        super(Model, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512*ResidualBlock.expansion, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = torch.flatten(out,1 )
        out = self.fc(out)
        return out

if __name__ == '__main__':
    resnet = Model(ResidualBlock,num_classes = 2)
    if torch.cuda.is_available():
        resnet.cuda()
    print(resnet)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(resnet.parameters(),lr = 0.01)

    train = get_dataset(train = True)

    for epoch in tqdm(range(10)):
        for i,(images,target) in enumerate(train):
            images = images.to(device)
            target = target.to(device)

            out = resnet(images)
            loss = criterion(out,target)
            print(loss)

            # Back-propogation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _,pred = torch.max(out.data,1)
            correct = (pred == target).sum().item()

            if i % 100 == 0:
                torch.save(resnet.state_dict(),"model")
                print(f" epoch: {epoch}\tloss: {loss.data}\tAccuracy: {(correct/target.size(0)) * 100}%")

    test = get_data(train = False)

    with torch.no_grad():
        correct = 0
        total = 0
        for i,(images,target) in tqdm(enumerate(test)):
            images = images.to(device)
            target = target.to(device)

            out = resnet(images)
            _,pred = torch.max(out.data,1)
            total += target.size(0)
            correct += (pred == target).sum().item()
        print(f"Accuracy: {(correct/total) * 100}")
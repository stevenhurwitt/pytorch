""" basic neural net run w/ pytorch on MNIST data """
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_data(batch_size = 200,train = True):
    data = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=train, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    return(data)


class Model(nn.Module):

    def __init__(self):
        super(Model,self).__init__()
        self.l1 = nn.Sequential(
                nn.Conv2d(1,32,kernel_size = 5,stride = 1,padding = 2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2, stride = 2)
                )
        self.l2 = nn.Sequential(
                nn.Conv2d(32,64,kernel_size = 5,stride = 1,padding = 2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2, stride = 2)
                )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7*7*64,1000)
        self.fc2 = nn.Linear(1000,10)

    def forward(self,x):
            x = self.l1(x)
            x = self.l2(x)
            x = x.reshape(x.size(0),-1)
            x = self.drop_out(x)
            x = self.fc1(x)
            x = self.fc2(x)
            x = x.reshape(x.size(0),-1)

            return(x)


if __name__  == '__main__':

    cnn = Model()
    if torch.cuda.is_available():
        cnn.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(),lr = 0.01)

    train = get_data()

    for epoch in tqdm(range(10)):
        for i,(images,target) in enumerate(train):
            images = images.to(device)
            target = target.to(device)

            out = cnn(images)
            loss = criterion(out,target)

            # Back-propogation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _,pred = torch.max(out.data,1)
            correct = (pred == target).sum().item()

            if i % 100 == 0:
                print(f" epoch: {epoch}\tloss: {loss.data}\tAccuracy: {(correct/target.size(0)) * 100}%")


    test = get_data(train = False)

    with torch.no_grad():
        correct = 0
        total = 0
        for i,(images,target) in tqdm(enumerate(test)):
            images = images.to(device)
            target = target.to(device)

            out = cnn(images)
            _,pred = torch.max(out.data,1)
            total += target.size(0)
            correct += (pred == target).sum().item()
        print(f"Accuracy: {(correct/total) * 100}")


        
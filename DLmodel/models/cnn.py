import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, in_channels, n_class):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=128, kernel_size=7, padding=1)
        self.sigmoid1 = nn.ReLU()
        self.pool1 = nn.AvgPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7, padding=1)
        self.sigmoid2 = nn.ReLU()
        self.pool2 = nn.AvgPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.liner = nn.Linear(5888, 1024)
        self.relu = nn.ReLU()
        self.liner2 = nn.Linear(1024, n_class)
        self.softmax = nn.Softmax(dim=1)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.sigmoid1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.sigmoid2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.liner(x)
        x = self.relu(x)
        x = self.liner2(x)
        x = self.softmax(x)
        return x


def test():
    x = torch.randn(8, 1, 105)
    model = CNN(in_channels=1, n_class=2)
    y = model.forward(x)
    print(y.shape)    
    print(y)

if __name__ == '__main__':
    test()
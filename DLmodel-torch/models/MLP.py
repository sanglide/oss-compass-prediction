import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out, softmax=False):
        super(MLPBlock, self).__init__()
        self.drop = nn.Dropout(p=drop_out)
        self.layer = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.relu = nn.ReLU()
        self.softmax = None
        if softmax:
            self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.drop(x)
        x = self.layer(x)
        if self.softmax != None:
            x = self.softmax(x)
        else:
            x = self.relu(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, n_class) :
        super(MLP, self).__init__()
        self.Layer1 = MLPBlock(in_channels=in_channels, out_channels=500, drop_out=0.1)
        self.Layer2 = MLPBlock(in_channels=500, out_channels=500, drop_out=0.2)
        self.Layer3 = MLPBlock(in_channels=500, out_channels=500, drop_out=0.2)
        self.Layer4 = MLPBlock(in_channels=500, out_channels=n_class, drop_out=0.3, softmax=True)

    def forward(self, x):
        x = self.Layer1(x)
        x = self.Layer2(x)
        x = self.Layer3(x)
        x = self.Layer4(x)
        return x.squeeze()
    
def test():
    x = torch.randn(8, 1, 105)
    model = MLP(in_channels=105, n_class=2)
    y = model.forward(x)
    print(y.shape)
    print(y)

if __name__ == '__main__':
    test()

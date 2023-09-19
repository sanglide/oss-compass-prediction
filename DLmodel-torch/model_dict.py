from models.cnn import CNN
from models.ResNet import ResNet
from models.MLP import MLP

DLmodel_dict = {
    "MLP": MLP(in_channels=105, n_class=2),
    "CNN": CNN(in_channels=1, n_class=2),
    "ResNet": ResNet(in_channels=1, n_class=2, mid_channels=256),
}
from models.cnn import CNN
from models.MLP import MLP as myMLP
from models.ResNet import ResNet1D, ResidualBlock1D
from models.inception import InceptionModel
from tsai.all import *

DLmodel_dict = {
    "MLP": myMLP(in_channels=105, n_class=2),
    "CNN": CNN(in_channels=1, n_class=2),
    "ResNet": ResNet1D(ResidualBlock1D, [2, 2, 2]),
    "Inception": InceptionModel(),
    "tsai-MLP": MLP(c_in=1, c_out=2, seq_len=105),
    "tsai-FCN": FCN(c_in=1, c_out=2),
    "tsai-FCNPlus": FCNPlus(c_in=1, c_out=2),
    "tsai-ResNet": ResNet(c_in=1, c_out=2),
    "tsai-ResNetPlus": ResNetPlus(c_in=1, c_out=2),
    "tsai-xResNet":  xresnet1d34_deeper(c_in=1, c_out=2),
    "tsai-xResNetPlus": xresnet1d50_deeperplus(c_in=1, c_out=2),
    "tsai-ResCNN": ResCNN(c_in=1, c_out=2,coord=True, separable=True),
    "tsai-TCN": TCN(c_in=1, c_out=2),
    "tsai-InceptionTime": InceptionTime(c_in=1, c_out=2, seq_len=105),
    "tsai-InceptionTimePlus": InceptionTimePlus(c_in=1, c_out=2, seq_len=105),
    "tsai-XceptionTime": XceptionTime(c_in=1, c_out=2),
    "tsai-XceptionTimePlus": XceptionTimePlus(c_in=1, c_out=2),
    "tsai-OmniScaleCNN": OmniScaleCNN(c_in=1, c_out=2, seq_len=105),
    "tsai-XCM": XCM(c_in=1, c_out=2, seq_len=105, fc_dropout=.5),
    "tsai-XCMPlus": XCMPlus(c_in=1, c_out=2, seq_len=105, fc_dropout=0.5),
    "tsai-Transformer": TransformerModel (c_in=1, c_out=2, d_model=64, n_head=1, d_ffn=128,
                   dropout=0.1, activation='relu', n_layers=1),
    "tsai-TSiT": TSiTPlus(c_in=1, c_out=2, seq_len=105, attn_dropout=.1, dropout=.1, use_token=True),
    "tsai-RNN-FCN": RNN_FCNPlus(c_in=1, c_out=2, seq_len=105),
    "tsai-RNNPlus": RNNPlus(c_in=1, c_out=2),
    "tsai-LSTMPlus": LSTMPlus(c_in=1, c_out=2),
    "tsai-TransformerLSTMPlus": TransformerLSTMPlus(c_in=1, c_out=2, seq_len=105)
}
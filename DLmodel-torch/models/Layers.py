import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    """
    # 保证输入和输出的尺寸相同
    :param input 输入数据
    :param weight 卷积核的权重
    :param bias 偏置
    :param stride 步幅
    :param dilation 膨胀率
    :param groups 卷积分组数
    """
    # stride and dilation are expected to be tuples
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_in = l_out = input.size(2)
    # 计算填充量
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])
    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)

class Conv1dSamePadding(nn.Conv1d):

    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride, self.dilation, self.groups)
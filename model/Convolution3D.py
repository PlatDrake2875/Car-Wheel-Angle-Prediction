import torch
import torch.nn as nn
import torch.nn.functional as F

# !/usr/bin/env python
# coding: utf-8

import torch
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F


class Convolution3D(nn.Module):
    def __init__(self):
        super(Convolution3D, self).__init__()
        self.Convolution1 = nn.Conv3d(in_channels=3, out_channels=128, kernel_size=(3, 3, 3), stride=1,
                                      padding=(1, 0, 0), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.BatchN1 = nn.BatchNorm3d(num_features=128, affine=True, track_running_stats=True)

        self.MaxPooling1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0), dilation=1,
                                        return_indices=False, ceil_mode=False)
        self.MaxPooling2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0), dilation=1,
                                        return_indices=False, ceil_mode=False)

        self.Convolution2 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=(1, 0, 0))
        self.BatchN2 = nn.BatchNorm3d(num_features=64, affine=True, track_running_stats=True)

        self.ResConvolution1 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1, 1))
        self.averagePool1 = nn.AvgPool3d(kernel_size=1, stride=1, padding=(0, 0, 0))
        self.ResBatchN1 = nn.BatchNorm3d(num_features=64, affine=True,
                                         track_running_stats=True)

        self.Convolution3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 0, 0))
        self.BatchN3 = nn.BatchNorm3d(num_features=64, affine=True, track_running_stats=True)

        self.ResConvolution2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1, 1))
        self.averagePool2 = nn.AvgPool3d(kernel_size=1, stride=1, padding=(0, 0, 0))
        self.ResBatchN2 = nn.BatchNorm3d(num_features=64, affine=True,
                                         track_running_stats=True)

        self.Convolution4 = nn.Conv3d(in_channels=64, out_channels=8, kernel_size=(3, 3, 3), stride=1,
                                      padding=(1, 0, 0), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.BatchN4 = nn.BatchNorm3d(num_features=8, affine=True, track_running_stats=True)

        self.Convolution5 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(3, 3, 3), stride=1, padding=(1, 0, 0),
                                      dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.BatchN5 = nn.BatchNorm3d(num_features=8, affine=True, track_running_stats=True)

        self.Convolution6 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(3, 3, 3), stride=1, padding=(1, 0, 0),
                                      dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.BatchN6 = nn.BatchNorm3d(num_features=8, affine=True, track_running_stats=True)

        self.Flatten1 = nn.Flatten(start_dim=2)

        self.LSTM1 = nn.LSTM(input_size=10488, hidden_size=512, num_layers=1, batch_first=True)
        self.LSTM2 = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        self.LSTM3 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        self.LSTM4 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        self.LSTM5 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
        self.LSTM6 = nn.LSTM(input_size=32, hidden_size=16, num_layers=1, batch_first=True)

        self.fc1 = nn.Linear(in_features=16, out_features=512, bias=True)
        self.fc2 = nn.Linear(in_features=512, out_features=128, bias=True)
        self.fc3 = nn.Linear(in_features=128, out_features=64, bias=True)
        self.fc4 = nn.Linear(in_features=64, out_features=16, bias=True)
        self.fc5 = nn.Linear(in_features=16, out_features=1, bias=True)

    def forward(self, Input):
        # 3*3*3 3D Conv 3
        image = F.relu(self.BatchN1(self.Convolution1(Input)))
        # 3D Max Pooling
        image = self.MaxPooling1(image)
        # 3D Max Pooling
        image = self.MaxPooling2(image)
        # 3*3*3 3D Conv 128
        image = F.relu(self.BatchN2(self.Convolution2(image)))

        # ResNet - 3*3*3 3D Conv 64
        Residual = image
        Res_Output = F.relu(self.ResBatchN1(self.ResConvolution1(image)))
        image = F.relu(Residual + Res_Output)
        image = self.averagePool1(image)

        # 3*3*3 3D Conv 64
        image = F.relu(self.BatchN3(self.Convolution3(image)))

        # ResNet - 3*3*3 3D Conv 64
        Residual = image
        Res_Output = F.relu(self.ResBatchN2(self.ResConvolution2(image)))
        image = F.relu(Residual + Res_Output)

        image = self.averagePool2(image)

        # 3*3*3 3D Conv 8
        image = F.relu(self.BatchN4(self.Convolution4(image)))
        image = F.relu(self.BatchN5(self.Convolution5(image)))
        image = F.relu(self.BatchN6(self.Convolution6(image)))

        # LSTM 512
        image = image.permute([0, 2, 1, 3, 4])
        image = self.Flatten1(image)
        image = self.LSTM1(image)
        image = image[0]
        image = torch.tanh(image)

        # LSTM 256-128-64-32-16
        image = torch.tanh(self.LSTM2(image)[0])
        image = torch.tanh(self.LSTM3(image)[0])
        image = torch.tanh(self.LSTM4(image)[0])
        image = torch.tanh(self.LSTM5(image)[0])
        image = torch.tanh(self.LSTM6(image)[0])

        # FC 512
        image = image.permute(1, 0, 2)
        image = F.relu(self.fc1(image))

        # FC 128-64-16-1
        image = F.relu(self.fc2(image))
        image = F.relu(self.fc3(image))
        image = F.relu(self.fc4(image))
        angle = self.fc5(image)

        return angle


if __name__ == "__main__":
    model = Convolution3D()
    dummy_input = torch.randn(1, 3, 15, 120, 320)
    output = model(dummy_input)
    print("Output:", output)

import torch.nn as nn


class SegNet(nn.Module):
    def __init__(self, in_chan_num, out_chan_num):
        super(SegNet, self).__init__()
        self.encoder1 = Encorder2Convs(in_chan_num, 64)
        self.encoder2 = Encorder2Convs(64, 128)
        self.encoder3 = Encorder3Convs(128, 256)
        self.encoder4 = Encorder3Convs(256, 512)
        self.encoder5 = Encorder3Convs(512, 512)

        self.decoder1 = Encorder2Convs(512, 512)
        self.decoder2 = Encorder2Convs(512, 256)
        self.decoder3 = Encorder3Convs(256, 128)
        self.decoder4 = Encorder3Convs(128, 64)
        self.decoder5 = Encorder3Convs(64, out_chan_num)

    def forward(self, input):
        output, indices1 = self.encoder1(input)
        output, indices2 = self.encoder2(output)
        output, indices3 = self.encoder3(output)
        output, indices4 = self.encoder4(output)
        output, indices5 = self.encoder5(output)

        output = self.decoder1(output, indices1)
        output = self.decoder2(output, indices2)
        output = self.decoder3(output, indices3)
        output = self.decoder4(output, indices4)
        output = self.decoder5(output, indices5)

        return output


class Encorder2Convs(nn.Module):
    def __init__(self, in_chan_num, out_chan_num):
        super(Encorder2Convs, self).__init__()
        self.conv_batch_relu1 = ConvBatchReLU(in_chan_num, out_chan_num)
        self.conv_batch_relu2 = ConvBatchReLU(out_chan_num, out_chan_num)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, input):
        output = self.conv_batch_relu1(input)
        output = self.conv_batch_relu2(output)
        output, indices = self.pool(output)

        return output, indices


class Encorder3Convs(nn.Module):
    def __init__(self, in_chan_num, out_chan_num):
        super(Encorder3Convs, self).__init__()
        self.conv_batch_relu1 = ConvBatchReLU(in_chan_num, out_chan_num)
        self.conv_batch_relu2 = ConvBatchReLU(out_chan_num, out_chan_num)
        self.conv_batch_relu3 = ConvBatchReLU(out_chan_num, out_chan_num)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, input):
        output = self.conv_batch_relu1(input)
        output = self.conv_batch_relu2(output)
        output = self.conv_batch_relu3(output)
        output, indices = self.pool(output)

        return output, indices


class Decorder2Convs(nn.modules):
    def __init__(self, in_chan_num, out_chan_num):
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv_batch_relu1 = ConvBatchReLU(in_chan_num, out_chan_num)
        self.conv_batch_relu2 = ConvBatchReLU(out_chan_num, out_chan_num)

    def forward(self, input, indices):
        output = self.unpool(input, indices)
        output = self.conv_batch_relu1(output)
        output = self.conv_batch_relu2(output)
        output = nn.ReLU(output)

        return output


class Decorder3Convs(nn.modules):
    def __init__(self, in_chan_num, out_chan_num):
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv_batch_relu1 = ConvBatchReLU(in_chan_num, out_chan_num)
        self.conv_batch_relu2 = ConvBatchReLU(out_chan_num, out_chan_num)
        self.conv_batch_relu3 = ConvBatchReLU(out_chan_num, out_chan_num)

    def forward(self, input, indices):
        output = self.unpool(input, indices)
        output = self.conv_batch_relu1(output)
        output = self.conv_batch_relu2(output)
        output = self.conv_batch_relu3(output)
        output = nn.ReLU(output)

        return output


class ConvBatchReLU(nn.Module):
    def __init__(self, in_chan_num, out_chan_num):
        super(ConvBatchReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan_num, out_chan_num, kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_chan_num)

    def forward(self, input):
        output = self.conv(input)
        output = self.batch_norm(output)
        output = nn.ReLU(output)

        return output

#   VGG16_Atchitecture:
#     Sequential(
#   (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (1): ReLU(inplace)
#   (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (3): ReLU(inplace)
#   (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (6): ReLU(inplace)
#   (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (8): ReLU(inplace)
#   (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (11): ReLU(inplace)
#   (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (13): ReLU(inplace)
#   (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (15): ReLU(inplace)
#   (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (18): ReLU(inplace)
#   (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (20): ReLU(inplace)
#   (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (22): ReLU(inplace)
#   (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (25): ReLU(inplace)
#   (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (27): ReLU(inplace)
#   (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (29): ReLU(inplace)
#   (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
# )

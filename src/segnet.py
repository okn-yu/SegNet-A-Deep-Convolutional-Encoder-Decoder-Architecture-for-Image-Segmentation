import torch.nn as nn

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

class SegNet(nn.modules):
    def __init__(self, in_chan_num, out_chan_num):
        super().__init__(SegNet, self)
        self.encoder_network = self.encoders(in_chan_num)
        self.decoder_netwrok = self.decoders(out_chan_num)
        self.pixel_wise_classifier = nn.Softmax()

    def forward(self):
        pass

    def encoders(self, in_chan_num):
        pass

    def decoders(self, out_chan_num):
        pass

    def pixel_wise_classifier(self):
        pass


class Encorder2Convs(nn.modules):
    def __init__(self, in_chan_num, out_chan_num):
        super().__init__(Encorder2Convs, self)
        self.conv1 = nn.Conv2d(in_chan_num, out_chan_num, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_chan_num)
        self.conv2 = nn.Conv2d(out_chan_num, out_chan_num, kernel_size=3, stride=1, padding=1)
        self.batch_norum2 = nn.BatchNorm2d(out_chan_num)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, input):
        output = self.conv1(input)
        output = self.batch_norm1(output)
        output = nn.ReLU(output)
        output = self.conv2(output)
        output = self.batch_norum2(output)
        output = nn.ReLU(output)
        output, indices = self.pool(output)

        return output, indices


class Encorder3Convs(nn.modules):
    def __init__(self, in_chan_num, out_chan_num):
        super().__init__(Encorder3Convs, self)
        self.conv1 = nn.Conv2d(in_chan_num, out_chan_num, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_chan_num)
        self.conv2 = nn.Conv2d(out_chan_num, out_chan_num, kernel_size=3, stride=1, padding=1)
        self.batch_norum2 = nn.BatchNorm2d(out_chan_num)
        self.conv3 = nn.Conv2d(out_chan_num, out_chan_num, kernel_size=3, stride=1, padding=1)
        self.batch_norum3 = nn.BatchNorm2d(out_chan_num)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, input):
        output = self.conv1(input)
        output = self.batch_norm1(output)
        output = nn.ReLU(output)
        output = self.conv2(output)
        output = self.batch_norum2(output)
        output = nn.ReLU(output)
        output = self.conv3(output)
        output = self.batch_norm3(output)
        output, indices = self.pool(output)

        return output, indices

class Decorder2Convs(nn.modules):
    def __init__(self, in_chan_num, out_chan_num):
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_chan_num, out_chan_num, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_chan_num)
        self.conv2 = nn.Conv2d(out_chan_num, out_chan_num, kernel_size=3, stride=1, padding=1)
        self.batch_norum2 = nn.BatchNorm2d(out_chan_num)

    def forward(self, input, indices):
        output = self.unpool(input, indices)
        output = self.conv1(output)
        output = self.batch_norm1(output)
        output = nn.ReLU(output)


class Decorder3Convs(nn.modules):
    pass


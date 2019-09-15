
""" Задание конволюционного блока: Конволюция -> Нормализация -> функция активации """

def conv_bn_relu(in_planes, out_planes, kernel=3, stride=1, padding=1):
     net = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=1),
                         nn.BatchNorm2d(num_features=out_planes),
                         nn.ReLU(True))
     return net;

""" Сборка сети из конволюционных блоков, дропаутов и пулинга. Для референса использовалась модель U-net """
class SegmenterModel1(nn.Module):
    def __init__(self, in_size=3):
        super(SegmenterModel1, self).__init__()
        
        
        self.do = nn.Dropout(0.5)
        self.conv1_1 = conv_bn_relu(3, 64, kernel=3, stride=1, padding=1)
        self.conv1_2 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2_1 = conv_bn_relu(64, 128, kernel=3, stride=1, padding=1)
        self.conv2_2 = conv_bn_relu(128, 128, kernel=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv3_1 = conv_bn_relu(128, 256, kernel=3, stride=1, padding=1)
        self.conv3_2 = conv_bn_relu(256, 256, kernel=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv4_1 = conv_bn_relu(256, 512, kernel=3, stride=1, padding=1)
        self.conv4_2 = conv_bn_relu(512, 512, kernel=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv5_1 = conv_bn_relu(512, 1024, kernel=3, stride=1, padding=1)
        self.conv5_2 = conv_bn_relu(1024, 1024, kernel=3, stride=1, padding=1)
        self.up_conv1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv10_1 = conv_bn_relu(1024, 512, kernel=3, stride=1, padding=1)
        self.conv10_2 = conv_bn_relu(512, 512, kernel=3, stride=1, padding=1)
        self.up_conv3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv6_1 = conv_bn_relu(512, 256, kernel=3, stride=1, padding=1)
        self.conv6_2 = conv_bn_relu(256, 256, kernel=3, stride=1, padding=1)
        self.up_conv2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv7_1 = conv_bn_relu(256, 128, kernel=3, stride=1, padding=1)
        self.conv7_2 = conv_bn_relu(128, 128, kernel=3, stride=1, padding=1)
        self.up_conv3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv8_1 = conv_bn_relu(128, 64, kernel=3, stride=1, padding=1)
        self.conv8_2 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1)
        self.conv9_1 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1)
        self.conv9_3 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1)
        self.conv9_2 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(num_features=1),
                                nn.Sigmoid())
         
    def forward(self, input):
        output = self.conv1_1 (input)
        output = self.do (output)
        output = self.conv1_2 (output)
        output = self.do (output)
        output = self.pool1 (output)
        output = self.do (output)
        output = self.conv2_1 (output)
        output = self.do (output)
        output = self.conv2_2 (output)
        output = self.do (output)
        output = self.pool2 (output)
        output = self.do (output)
        output = self.conv3_1 (output)
        output = self.do (output)
        output = self.conv3_2 (output)
        output = self.do (output)
        output = self.pool3 (output)
        output = self.do (output)
        output = self.conv4_1 (output)
        output = self.do (output)
        output = self.conv4_2 (output)
        output = self.do (output)
        output = self.pool4 (output)
        output = self.do (output)
        output = self.conv5_1 (output)
        output = self.do (output)
        output = self.conv5_2 (output)
        output = self.do (output)
        output = self.up_conv1 (output)
        output = self.do (output)
        output = self.conv10_1 (output)
        output = self.do (output)
        output = self.conv10_2 (output)
        output = self.do (output)
        output = self.up_conv3 (output)
        output = self.do (output)
        output = self.conv6_1 (output)
        output = self.do (output)
        output = self.conv6_2 (output)
        output = self.do (output)
        output = self.up_conv2 (output)
        output = self.do (output)
        output = self.conv7_1 (output)
        output = self.do (output)
        output = self.conv7_2 (output)
        output = self.do (output)
        output = self.up_conv3 (output)
        output = self.do (output)
        output = self.conv8_1 (output)
        output = self.do (output)
        output = self.conv8_2 (output)
        output = self.do (output)
        output = self.conv9_1 (output)
        output = self.do (output)
        output = self.conv9_3 (output)
        output = self.do (output)
        output = self.conv9_2 (output)
        return output

import torch
import torch.nn as nn
import numpy as np

# to create the resnet layers
class block(nn.Module):

    # identity_downsample --> conv layer
    def __init__(self, input_channel, output_channel, identity_downsample=None, stride=1):
        super(block,self).__init__()

        # number of output channel is always 4 times the number of input channel in a block
        self.expansion = 4

        """ ---- 1st conv layer (kernel_size = 1) ---- """
        # 1st convolution layer
        self.conv1 = nn.Conv3d(in_channels=input_channel,
                               out_channels=output_channel,
                               kernel_size=1,
                               stride=1,
                               padding=0)

        # 1st batch norm layer
        self.bn1 = nn.BatchNorm3d(output_channel)

        """ ---- 2nd conv layer (kernel_size = 3)---- """
        # 2nd convolution layer
        # the stride is from the init
        # the in_channels will be the output from the previous layer
        self.conv2 = nn.Conv3d(in_channels=output_channel,
                               out_channels=output_channel,
                               kernel_size=3,
                               stride=stride,
                               padding=1)

        # 2nd batch norm layer
        self.bn2 = nn.BatchNorm3d(output_channel)

        """ ---- 3rd layer conv (kernel_size = 1)---- """
        # 3rd convolution layer
        # the output channel will be 4 times the number of the input channel from the previous layer
        # the in_channels will be the output from the previous layer
        self.conv3 = nn.Conv3d(in_channels=output_channel,
                               out_channels=output_channel * self.expansion,
                               kernel_size=1,
                               stride=1,
                               padding=0)

        # 3rd batch norm layer
        self.bn3 = nn.BatchNorm3d(output_channel * self.expansion)

        """ ---- ReLU layer ---- """
        self.relu = nn.ReLU()

        """ ---- identity mapping ----"""
        # conv layer that do the identity mapping
        # to ensure same shape in the later layers
        # If their sizes mismatch, then the input goes into an identity
        # this is for the skipped connection
        self.identity_downsample = identity_downsample

    # forward pass
    def forward(self, x):
        identity = x


        # A basic ResNet block is composed by two layers of 3x3 conv/batchnorm/relu
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        # use the identity downsample if there is a need to change the shape
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        # current output plus the residual skipped connection
        x = x + identity
        x = self.relu(x)
        return x

# [3,4,6,3] --> layers per block
class ResNet(nn.Module):
    # block --> from the block class
    # layers --> number of times to use the block class [3,4,6,3]
    # image_channels --> number of channels of the input (normally is 3, RGB)
    # num_classes --> number of classes in the data (remove it for CVLR)
    def __init__(self, block, layers, image_channels):
        super(ResNet,self).__init__()

        # first layer
        self.input_channel = 64
        # kernel size is 5 as mentioned in the paper
        # original kernel size is 7
        self.conv1 = nn.Conv3d(in_channels=image_channels, out_channels=64, kernel_size=5, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()

        # max pooling layer
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Resnet layers
        # out_channels * 4 at the end
        # all data layers are stride 2 as mentioned in the paper
        #self.layer1 = self._make_layers(block, layers[0], out_channels=64, stride=1)
        self.layer1 = self._make_layers(block, layers[0], out_channels=64, stride=2)
        self.layer2 = self._make_layers(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layers(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layers(block, layers[3], out_channels=512, stride=2)

        # the features
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))

        # fc layer
        #self.fc = nn.Linear(512 * 4, num_classes)

        # MLP layer
        # output channel * 4 from the layer 4
        self.l1 = nn.Linear(512*4, 512*4)
        self.l2 = nn.Linear(512*4, 128)


    # get the features from the CNN layers
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # get the correct shape to make the output 1x1
        """ Feature extraction """
        # to be use for downstream tasks
        h = self.avgpool(x)

        # reshape to send to the fully connected layer
        #h = x.reshape(h.shape[0], -1)

        """ MLP layer """
        h = h.squeeze()

        # Projection --> (Dense --> Relu --> Dense)
        x = self.l1(h)
        x = self.relu(x)
        x = self.l2(x)

        # original resnet FC
        #x = self.fc(x)

        return h, x


    # create ResNet Layers
    # num_residual_blocks --> number of times the block class will be used
    # out_channels --> number of channels for the output of the layer
    # calls the block multiple times
    def _make_layers(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        # layers that changes the number of channels for each input channel in subsequent blocks
        layers = []

        # if the stride changes or the input channel into the nxt block changes
        if stride != 1 or self.input_channel != out_channels * 4:
            # only want to change the channel so the kernel size will remain as 1
            identity_downsample = nn.Sequential(nn.Conv3d(in_channels = self.input_channel,
                                                          out_channels=out_channels*4,
                                                          kernel_size = 1,
                                                          stride = stride),
                                                nn.BatchNorm3d(out_channels * 4))

        # first block --> the only changes in stride and channels
        # out_channels will be multiplied by 4 at the end of each block
        # identity mapping is the addition of the skipped connection and the output of the layer
        # need to do identity downsample due to the difference in input channel in the first layer and output layer in the first block to do identity mapping
        # it will downsample the identity via passed convolution layer to successfully perform addition
        layers.append(block(self.input_channel, out_channels, identity_downsample, stride))

        # need to change the number of input channels to match the output channels of the previous block
        self.input_channel = out_channels * 4

        # -1 because one residual block have been calculated in
        # 'layers.append(block(self.input_channel, out_channels, identity_downsample, stride))' that changes the num of channels
        for i in range(num_residual_blocks - 1):
            # out_channels will be 256 after the end of the first block
            # for this first layer, the in_channels will be 256 and the out_channels will be 64
            # therefore, need to map 256 (in_channels) to 64 (out_channels) --> at the end of the block, 64 * 4 = 256 again
            # stride will be one as well
            layers.append(block(self.input_channel, out_channels))

        # unpack the list of layers and pytorch will know that each layer will come after each other
        return (nn.Sequential(*layers))


def ResNet_3D_50(img_channels = 3, num_classes=1000):
    return ResNet(block, layers=[3,4,6,3], image_channels=img_channels)


def ResNet_3D_101(img_channels = 3, num_classes=1000):
    return ResNet(block, layers=[3,4,23,3], image_channels=img_channels)


def ResNet_3D_152(img_channels = 3, num_classes=1000):
    return ResNet(block, layers=[3,8,36,3], image_channels=img_channels)

def test():
    model = ResNet_3D_50()
    x = torch.randn(10, 3, 3, 224, 224)
    # get the representations and the projections
    ris, zis = model(x)
    #print(y.shape)
    print(ris.shape, zis.shape)

#test()


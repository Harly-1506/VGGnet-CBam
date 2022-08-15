import numpy as np

import torch
import torch.nn as nn
# import torch.nn.functional as F

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from models.cbam import *


#config for all VGGNet models
vgg11_config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

vgg13_config = [ 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512,
                512, 'M']


vgg16_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
                'M', 512, 512, 512, 'M']

vgg19_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512,
                512, 512, 'M', 512, 512, 512, 512, 'M']


class BasicBlock(nn.Module):
  def __init__(self, in_channels, out_channels, use_cbam = True, vgg_net = "vgg16"):
    super(BasicBlock, self).__init__()

    self.vgg_net = vgg_net
    self.cfg = out_channels
    self.use_cbam = use_cbam

    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu1 = nn.ReLU(inplace = True)

    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.relu2 = nn.ReLU(inplace = True)

    if self.cfg > 128 and self.vgg_net == "vgg16":
      self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
      self.bn3 = nn.BatchNorm2d(out_channels)
      self.relu3 = nn.ReLU(inplace = True)
    elif self.cfg >128 and self.vgg_net == "vgg19":
      self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
      self.bn4 = nn.BatchNorm2d(out_channels)
      self.relu4 = nn.ReLU(inplace = True)

    self.pooling = nn.MaxPool2d(kernel_size = 2)
  
    if self.use_cbam:
      self.cbam = CBAM(out_channels)

  def forward(self, x):

      # residual = x
      out = self.conv1(x)
      out = self.bn1(out)
      out = self.relu1(out)

      out = self.conv2(out)
      out = self.bn2(out)
      out = self.relu2(out)
      
      if self.cfg > 128 and self.vgg_net == "vgg16":
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pooling(out)
      elif self.cfg > 128 and self.vgg_net == "vgg_19":
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)
        out = self.pooling(out)
      else:
        out = self.pooling(out)

      if self.use_cbam:
        out = self.cbam(out)

      # out += residual
      out = self.relu1(out)

      return out



class VGGnet(nn.Module):
    def __init__(self,  block, network_name = "vgg16"):
        super(VGGnet, self).__init__()
        
        # self.features = features
        self.network_name = network_name

        self.layer1 = self._make_layers(BasicBlock, 3, 64, use_cbam = True, vgg_net = self.network_name)
        self.layer2 = self._make_layers(BasicBlock,64, 128,  use_cbam = True, vgg_net = self.network_name)
        self.layer3 = self._make_layers(BasicBlock, 128, 256,  use_cbam = True, vgg_net = self.network_name)
        self.layer4 = self._make_layers(BasicBlock, 256, 512, use_cbam = True, vgg_net = self.network_name)
        self.layer5 = self._make_layers(BasicBlock, 512, 512, use_cbam = True, vgg_net = self.network_name)

        self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1000)
        )

        self._initialize_weights()
        
    def _make_layers(self, block,in_channel, out_channel, use_cbam = True, vgg_net = "vgg16"):
        layers  = []

        layers.append(block(in_channel, out_channel, use_cbam = use_cbam, vgg_net = vgg_net))
        
        return nn.Sequential(*layers)



    def forward(self, x):
        # x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# def load_vgg(url_net, config, batch_norm, pretrained, progress):

#     model = VGGnet(BasicBlock)

#     return model

def vgg16_cbam(num_classes):

    model = VGGnet(BasicBlock, network_name = "vgg16")

    model.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, num_classes),
    )
    return model


def vgg19_cbam(num_classes):

    model = VGGnet(BasicBlock, network_name = "vgg19")

    model.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, num_classes),
    )
    return model
'''
Author : Stefano Malacrino stefano.malacrino@nds.ox.ac.uk
'''

from .unet_base import ConvBlock

from torchvision.models.densenet import DenseNet, model_urls, _Transition#, _load_state_dict
from torchvision.models.utils import load_state_dict_from_url

import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
import re

densenet_params = {
        'densenet121' : (32, (6, 12, 24, 16), 64),
        'densenet161' : (48, (6, 12, 36, 24), 96),
        'densenet169' : (32, (6, 12, 32, 32), 64),
        'densenet201' : (32, (6, 12, 48, 32), 64)
        }

class _DenseUNetEncoder(DenseNet):
    def __init__(self, skip_connections, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0):
        super(_DenseUNetEncoder, self).__init__(32, block_config, num_init_features, bn_size, drop_rate)
        
        self.skip_connections = skip_connections
        #remove norm5
        self.features = nn.Sequential(OrderedDict(list(self.features.named_children())[:-1]))
        delattr(self, 'classifier')
        
        for module in self.features.modules():
            if isinstance(module, nn.AvgPool2d) or isinstance(module, nn.MaxPool2d):
                module.register_forward_hook(lambda _, input, output : self.skip_connections.append(input[0]))

    def forward(self, x):
        return self.features(x)
        
class _DenseUNetDecoder(DenseNet):
    def __init__(self, skip_connections, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0):
        super(_DenseUNetDecoder, self).__init__(32, block_config, num_init_features, bn_size, drop_rate)
        
        self.skip_connections = skip_connections
        delattr(self, 'classifier')
        features = list(self.features.named_children())[4:-2]
        n = 1 
        for i in range(len(features)):
            name, module = features[i]
            if isinstance(module, _Transition):
                conv1 = (i < len(features) - 1)
                features[i] = (name, _TransitionUp(num_init_features*2**(n+1+conv1),num_init_features*2**(n-1), self.skip_connections, conv1))
                n += 1
        features.insert(0, ('transition0', _TransitionUp(num_init_features*4,num_init_features, self.skip_connections)))

        features.reverse()
        
        self.features = nn.Sequential(OrderedDict(features))
        self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
        self.features.add_module('relu0', nn.ReLU(inplace=True))
        
        self.final_upsample = nn.Sequential()
        self.final_upsample.add_module('conv0', nn.Conv2d(num_init_features, num_init_features, kernel_size=1, stride=1, bias=False))
        self.final_upsample.add_module('norm0', nn.BatchNorm2d(num_init_features))
        self.final_upsample.add_module('relu0', nn.ReLU(inplace=True))

    def forward(self, x, size):
        x = self.features(x)
        x = F.interpolate(x, size, mode='bilinear')
        return self.final_upsample(x)
          
        
class _TransitionUp(nn.Module):
    def __init__(self, num_input_features, num_output_features, skip_connections, conv1=True):
        super(_TransitionUp, self).__init__()
        
        self.skip_connections = skip_connections
        self.block0 = nn.Sequential()
        self.block0.add_module('norm0', nn.BatchNorm2d(num_input_features))
        self.block0.add_module('relu0', nn.ReLU(inplace=True))
        
        if conv1:
            self.block0.add_module('conv1', nn.Conv2d(num_input_features, num_input_features // 2,
                                              kernel_size=1, stride=1, bias=False))
            self.block0.add_module('norm1', nn.BatchNorm2d(num_input_features // 2))
            self.block0.add_module('relu1', nn.ReLU(inplace=True))
            num_input_features = num_input_features // 2
        
        self.block1 = nn.Sequential()
        self.block1.add_module('conv2', nn.Conv2d(num_input_features, num_input_features // 2,
                                          kernel_size=1, stride=1, bias=False))
        self.block1.add_module('norm2', nn.BatchNorm2d(num_input_features // 2))
        self.block1.add_module('relu2', nn.ReLU(inplace=True))
        
        self.block2 = nn.Sequential()
        self.block2.add_module('conv3', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        
    def forward(self, x):
        x = self.block0(x)
        x = F.interpolate(x, list(self.skip_connections[-1].shape[2:]), mode='bilinear')
        x = self.block1(x)
        x = torch.cat([x, self.skip_connections.pop()], 1)
        x = self.block2(x)
        return x


def _load_state_dict(model, model_url):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = load_state_dict_from_url(model_url, progress=True)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict, strict=False)
    
def _densenet_encoder(arch, skip_connections, growth_rate, block_config, num_init_features, pretrained, **kwargs):
    model = _DenseUNetEncoder(skip_connections, growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch])
    
    return model

def _densenet_decoder(skip_connections, growth_rate, block_config, num_init_features, **kwargs):
    return _DenseUNetDecoder(skip_connections, growth_rate, block_config, num_init_features, **kwargs)




class DenseUNet(nn.Module):
    def __init__(self, n_inputs, n_outputs, backbone_name = 'densenet121', pretrained_enconder = True, return_feat_maps = False, **kwargs):
        super(DenseUNet, self).__init__()
        self.skip_connections = []
        
        
        ## initial
        self.adaptor = nn.Sequential(
                            ConvBlock(n_inputs, 64, kernel_size = 7, batchnorm = True),
                            ConvBlock(64, 3, kernel_size = 3, batchnorm = True)
                        )
        
        #currently only works for densenet121...
        params = densenet_params[backbone_name]
        self.encoder = _densenet_encoder(backbone_name, self.skip_connections, *params, pretrained = pretrained_enconder, **kwargs)
        self.decoder = _densenet_decoder(self.skip_connections, *params, **kwargs)
        self.classifier = nn.Conv2d(64, n_outputs, kernel_size=1, stride=1, bias=True)
        
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        
        
    def forward(self, x):
        
        #if x.shape[1] != 3:
        x = self.adaptor(x)
        
        size = list(x.shape[2:])
        
        x = self.encoder(x)
        x = self.decoder(x, size)
        x = self.classifier(x)
        return x
    

if __name__ == '__main__':
    
    X = torch.rand((1, 3, 100, 100))
    #feats = backbone(X)
    
    model = DenseUNet(3, 3, backbone_name = 'densenet121')
    xout = model(X)
    
    assert xout.shape[2:] == X.shape[2:]
    
    loss = ((xout - X)**2).mean()
    loss.backward()
        

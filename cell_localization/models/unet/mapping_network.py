from .unet_base import unet_constructor, unet_input_halved
from .unet_attention import unet_attention
from .unet_squeeze_excitation import unet_squeeze_excitation
from .EDSR import EDSRBody
from .unet_resnet import UnetResnet
from .unet_densenet import UnetDensenet
from .unet_n2n import UNetN2N
from .dense_unet import DenseUNet

def get_mapping_network(n_inputs, n_ouputs, model_type = 'unet-simple',  **argkws):
    
    if model_type == 'unet-simple':
        constructor = unet_constructor
    elif model_type == 'unet-attention':
        constructor = unet_attention 
    elif model_type == 'unet-SE':
        constructor = unet_squeeze_excitation
    elif model_type == 'unet-input-halved':
        constructor = unet_input_halved
    elif model_type == 'unet-n2n':
        constructor = UNetN2N
    elif model_type == 'resnet':
        constructor = UnetResnet
    elif model_type == 'densenet':
        constructor = UnetDensenet
    elif model_type == 'dense-unet':
        constructor = DenseUNet
    elif model_type == 'EDSR':
        constructor = EDSRBody
    else:
        raise ValueError(f'Not implemented {model_type}')
    
    model = constructor(n_inputs, n_ouputs, **argkws)
    model.__name__ = model_type
    return model
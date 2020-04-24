from networks.segmentation_models_pytorch import *
from networks.mitmodels import UNet16, UNet11, LinkNet34
from networks.mitmodels import UNet as UNetS
from networks.efficient_segmentation_models import ESPNet, ICNet, DABNet

model_list = {'UNet': Unet,
              'UNetM': UnetMulti,
'UNetS': UNetS,
                'UNet11': UNet11,
                'UNet16': UNet16,
'LinkNet34': LinkNet34,
               'DeepLabV3': DeepLabV3,
               'PSPNet': PSPNet,
              'ESPNet': ESPNet,
              'ICNet': ICNet,
              'DABNet': DABNet}


def get_model(model_name):
    if model_name not in model_list.keys():
        print('Not valid model!')
        return None
    else:
        return model_list[model_name]

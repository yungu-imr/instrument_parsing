from networks.segmentation_models_pytorch import *

model_list = {'UNet': Unet,
               'DeepLabV3': DeepLabV3,
               'PSPNet': PSPNet}


def get_model(model_name):
    if model_name not in model_list.keys():
        print('Not valid model!')
        return None
    else:
        return model_list[model_name]

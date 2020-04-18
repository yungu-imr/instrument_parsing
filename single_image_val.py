from utils.options import Options
from pathlib import Path
import cv2
from networks.segmentation_models_pytorch import *
import torch
from torch.nn.Module import load_state_dict
import numpy as np
from albumentations.pytorch.functional import img_to_tensor

image_num = '000'
video_num = '3'
problem_type = 'binary_masks'
model_name = '/model_checkpoints/UNet_binary_fold0_20200414_004251/latest_net_onestage_parse_net_UNet_fold0.pth'
data_path = Path('/opt/dataset/instrument_segmentation/endovis2017/data/cropped_train')
factor = 255

image_path = data_path / 'instrument_dataset_{}'.format(video_num)
image_name = 'frame{}.png'.format(image_num)

def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    val_image_path = image_path / 'images' / image_name
    val_mask_path = image_path / problem_type / image_name

    val_image = img_to_tensor(load_image(val_image_path))
    val_mask = torch.from_numpy(load_image(val_mask_path)/factor.astype(np.uint8)).long()

    model = Unet(in_channels=3, classes=2)
    model.load_state_dict(torch.load(model_name))




import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import data.utils.prepare_data as prepare_data
from utils.options import Options
from albumentations.pytorch.functional import img_to_tensor
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose,
    PadIfNeeded,
    RandomCrop,
    CenterCrop
)


class RoboticsDataset(Dataset):
    def __init__(self, file_names, transform=None, problem_type='binary'):
        self.file_names = file_names
        self.transform = transform
        self.problem_type = problem_type

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)
        mask = load_mask(img_file_name, self.problem_type)
        contour = load_contour(img_file_name)

        data = {"image": image, "mask": mask, 'contour': contour}

        if self.transform is not None:
            augmented = self.transform(**data)
            image, mask,contour = augmented["image"], augmented["mask"], augmented['contour']

        if self.problem_type == 'binary':
            return img_to_tensor(image), torch.from_numpy(mask).long(), torch.from_numpy(contour).long()
        else:
            return img_to_tensor(image), torch.from_numpy(mask).long(), torch.from_numpy(contour).long()


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_contour(path):
    contour = cv2.imread(str(path).replace('images', 'parts_contours2').replace('jpg', 'png'), 0)
    return contour


def load_mask(path, problem_type):
    if problem_type == 'binary':
        mask_folder = 'binary_masks'
        factor = prepare_data.binary_factor
    elif problem_type == 'parts':
        mask_folder = 'parts_masks'
        factor = prepare_data.parts_factor
    elif problem_type == 'instruments':
        factor = prepare_data.instrument_factor
        mask_folder = 'instruments_masks'

    mask = cv2.imread(str(path).replace('images', mask_folder).replace('jpg', 'png'), 0)


    return (mask / factor).astype(np.uint8)


def get_train_dataloader(file_list, opt):
    data_transform = Compose([
            PadIfNeeded(min_height=opt.train_crop_height, min_width=opt.train_crop_width, p=1),
            RandomCrop(height=opt.train_crop_height, width=opt.train_crop_width, p=1),
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            Normalize(p=1)
        ], p=1)
    train_dataset = RoboticsDataset(file_names=file_list,
                                    transform=data_transform,
                                    problem_type=opt.problem_type)
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=True)
    return train_dataloader


def get_val_dataloader(file_list, opt):
    data_transform = Compose([
            PadIfNeeded(min_height=opt.val_crop_height, min_width=opt.val_crop_width, p=1),
            CenterCrop(height=opt.val_crop_height, width=opt.val_crop_width, p=1),
            Normalize(p=1)
        ], p=1)
    val_dataset = RoboticsDataset(file_names=file_list,
                                    transform=data_transform,
                                    problem_type=opt.problem_type)
    val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, pin_memory=True)
    return val_dataloader


if __name__ == '__main__':
    # Test code for dataloader

    options = Options()
    train_files, test_files = prepare_data.get_split(0)
    get_train_dataloader(train_files, options.opt)

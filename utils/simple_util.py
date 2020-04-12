from albumentations import Normalize
import torch
from data.utils.prepare_data import class_color
normalize = Normalize()
from PIL import Image
import inspect, re
import numpy as np
import os
import collections


def tensor2image(tensor):
    image_numpy = tensor[0].cpu().float().numpy().transpose((1, 2, 0))
    mean = np.array(normalize.mean, dtype=np.float32)
    mean *= normalize.max_pixel_value

    std = np.array(normalize.std, dtype=np.float32)
    std *= normalize.max_pixel_value
    # denominator = np.reciprocal(std, dtype=np.float32)
    image_numpy *= std
    image_numpy += mean
    return image_numpy.astype(np.uint8)


def get_predictions(output_batch):
    bs,c,h,w = output_batch.size()
    tensor = output_batch.data
    values, indices = tensor.cpu().max(1)
    indices = indices.view(bs,h,w)
    return indices


def tensor2mask(tensor, num_classes=2):
    if len(tensor.size()) == 4:
        tensor = get_predictions(tensor)
    # b,h,w = tensor.size()
    temp = tensor[0].cpu().float().numpy().astype(np.uint8)
    h,w = temp.shape
    full_mask = np.zeros((h, w, 3))
    for mask_label, sub_color in enumerate(class_color):
        full_mask[temp == mask_label, 0] = sub_color[2]
        full_mask[temp == mask_label, 1] = sub_color[1]
        full_mask[temp == mask_label, 2] = sub_color[0]
    return full_mask.astype(np.uint8)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

import torch.nn as nn
from torch.nn import functional as F
import torch
import numpy as np


def convert2onehot(tensor, num_classes):
    n, h, w = tensor.size()
    one_hot = torch.zeros(n, num_classes, h, w).cuda().scatter_(1, tensor.view(n, 1, h, w), 1)
    return one_hot


class BoundaryLoss(nn.Module):
    def __init__(self, boundry_kernel_size=3):
        super().__init__()
        self.boundary_kernel_size = boundry_kernel_size

    def forward(self, outputs_countour, outputs_mask, contour_encoding=None):
        """

        :param outputs_countour:
        :param outputs_mask:
        :return:
        """
        n, c, _, _ = outputs_mask.shape

        if contour_encoding is None:
            contour_encoding = []
            for i in range(0, c-2):
                for j in range(1, c-1):
                    contour_encoding.append([i,j])

        # softmax so that predicted map can be distributed in [0, 1]
        outputs_mask = F.softmax(outputs_mask, dim=1)
        boundary_est = F.max_pool2d(
            -outputs_mask + 1.0, kernel_size=self.boundary_kernel_size, stride=1, padding=(self.boundary_kernel_size - 1) // 2)
        boundary_est -= 1 - outputs_mask
        boundary_est = boundary_est.view(n, c, -1)

        outputs_countour = torch.softmax(outputs_countour, dim=1)
        n, c, _, _ = outputs_countour.shape
        outputs_countour = outputs_countour.view(n, c, -1)

        # Precision, Recall
        loss = 0

        for idx, class_pair in enumerate(contour_encoding):
            loss += F.kl_div(outputs_countour[:,idx+1,:],
                             boundary_est[:,class_pair[0],:]*boundary_est[:,class_pair[1],:])
        return loss


class CrossEntropy2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.nll_loss = nn.NLLLoss()

    def __call__(self, outputs, targets):
        if len(targets.size()) == 4:
            targets = torch.argmax(targets, dim=1)
        outputs = F.log_softmax(outputs, dim=1)
        loss = self.nll_loss(outputs, targets)
        return loss


class _DiceLoss(nn.Module):
    def __init__(self, smooth=1e-3):
        super(_DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        batch_size = target.size(0)

        pred_flat = pred.view(batch_size, -1)
        target_flat = target.view(batch_size, -1)

        intersection = pred_flat * target_flat

        loss = 2 * (intersection.sum(1) + self.smooth) / (pred_flat.sum(1) + target_flat.sum(1) + self.smooth)
        loss = 1 - loss.sum() / batch_size

        return loss


class DiceLoss(nn.Module):
    def __init__(self, num_classes=2, class_weight=None):
        super(DiceLoss, self).__init__()
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.dice_metric = _DiceLoss()

    def forward(self, pred, target):
        '''

        :param pred: B*C*H*W
        :param target: B*H*W or B*C*H*W
        :return: dice_metric
        '''

        assert self.num_classes == pred.size(1)

        if len(target.size()) == 3:
            # Convert it to onehot
            target_onehot = convert2onehot(target, self.num_classes)
        else:
            target_onehot = target

        pred = F.softmax(pred, dim=1)
        # dice_all = []
        loss = 0
        for i in range(1, self.num_classes): #ignore background
            current_dice = self.dice_metric(pred[:, i], target_onehot[:, i])
            if self.class_weight is not None:
                current_dice *= self.class_weight[i]
            loss += current_dice
            # dice_all.append(current_dice)
        return loss


class JaccardLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6):
        super(JaccardLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred, target):
        '''
          :param pred: B*C*H*W
          :param target: B*H*W or B*C*H*W
          :return: jaccard
        '''

        batch_size = pred.size(0)

        pred = F.softmax(pred, dim=1)

        if len(target.size()) == 3:
            # Convert it to onehot
            target_onehot = convert2onehot(target, self.num_classes)
        else:
            target_onehot = target

        # Numerator Product
        inter = pred * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(batch_size, self.num_classes, -1).sum(2)

        # Denominator
        union = pred + target_onehot - (pred * target_onehot)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(batch_size, self.num_classes, -1).sum(2)

        loss = inter / (union + self.smooth)

        # Return average loss over classes and batch
        return -loss.mean()


class CEDiceLoss(nn.Module):
    def __init__(self, num_classes=2, class_weight=None):
        super(CEDiceLoss, self).__init__()
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.dice_metric = DiceLoss(num_classes=num_classes)
        self.crossentropy = CrossEntropy2D()

    def forward(self, pred, target):
        '''

        :param pred: B*C*H*W
        :param target: B*H*W or B*C*H*W
        :return: dice_metric
        '''
        dice_loss = self.dice_metric(pred, target)
        ce_loss = self.crossentropy(pred, target)

        return ce_loss + dice_loss


def general_dice(y_true, y_pred, num_classes=2):
    '''

    :param y_true: B*H*W ndarray
    :param y_pred: B*C*H*W ndarray
    :param num_classes:
    :return:
    '''
    result = []
    if len(y_pred.shape) == 4:
        y_pred = y_pred.argmax(axis=1)
    for instrument_id in range(1, num_classes):  # ignore background

        if np.all(y_true != instrument_id):
            if np.all(y_pred != instrument_id):
                result += [1]
            else:
                result += [0]
        else:
            result += [dice(y_true == instrument_id, y_pred == instrument_id)]

    return np.mean(result)


def general_jaccard(y_true, y_pred, num_classes=2):
    '''

    :param y_true:
    :param y_pred:
    :param num_classes:
    :return:
    '''
    result = []
    if len(y_pred.shape) == 4:
        y_pred = y_pred.argmax(axis=1)
    for instrument_id in range(1, num_classes):  # ignore background
        if np.all(y_true != instrument_id):
            if np.all(y_pred != instrument_id):
                result += [1]
            else:
                result += [0]
        else:
            result += [jaccard(y_true == instrument_id, y_pred == instrument_id)]

    return np.mean(result)


def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)


def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
    replace_indices = np.vstack((
        ground_truth.flatten(),
        prediction.flatten())
    ).T
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(nr_labels, nr_labels),
        range=[(0, nr_labels), (0, nr_labels)]
    )
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix


def calculate_iou(confusion_matrix):
    ious = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = true_positives + false_positives + false_negatives
        if denom == 0:
            iou = 0
        else:
            iou = float(true_positives) / denom
        ious.append(iou)
    return ious


def calculate_dice(confusion_matrix):
    dices = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            dice = 0
        else:
            dice = 2 * float(true_positives) / denom
        dices.append(dice)
    return dices


class ICNetLoss(nn.CrossEntropyLoss):
    """Cross Entropy Loss for ICNet"""

    def __init__(self, aux_weight=0.4, ignore_index=0):
        super(ICNetLoss, self).__init__(ignore_index=ignore_index)
        self.aux_weight = aux_weight

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])

        pred, pred_sub4, pred_sub8, pred_sub16, target = tuple(inputs)
        # [batch, H, W] -> [batch, 1, H, W]
        target = target.unsqueeze(1).float()
        target_sub4 = F.interpolate(target, pred_sub4.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        target_sub8 = F.interpolate(target, pred_sub8.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        target_sub16 = F.interpolate(target, pred_sub16.size()[2:], mode='bilinear', align_corners=True).squeeze(
            1).long()
        loss1 = super(ICNetLoss, self).forward(pred_sub4, target_sub4)
        loss2 = super(ICNetLoss, self).forward(pred_sub8, target_sub8)
        loss3 = super(ICNetLoss, self).forward(pred_sub16, target_sub16)
        # return dict(loss=loss1 + loss2 * self.aux_weight + loss3 * self.aux_weight)
        return loss1 + loss2 * self.aux_weight + loss3 * self.aux_weight
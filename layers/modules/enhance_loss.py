from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim
from data.config import cfg

def gradient(input_tensor, direction):
    smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
    smooth_kernel_y = torch.transpose(smooth_kernel_x, 2, 3)

    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    grad_out = torch.abs(F.conv2d(input_tensor, kernel,
                                  stride=1, padding=1))
    return grad_out

def ave_gradient(input_tensor, direction):
    return F.avg_pool2d(gradient(input_tensor, direction),
                        kernel_size=3, stride=1, padding=1)

def smooth(input_I, input_R):
    input_R = 0.299 * input_R[:, 0, :, :] + 0.587 * input_R[:, 1, :, :] + 0.114 * input_R[:, 2, :, :]
    input_R = torch.unsqueeze(input_R, dim=1)
    return torch.mean(gradient(input_I, "x") * torch.exp(-10 * ave_gradient(input_R, "x")) +
                      gradient(input_I, "y") * torch.exp(-10 * ave_gradient(input_R, "y")))

class EnhanceLoss(nn.Module):
    def __init__(self):
        super(EnhanceLoss, self).__init__()

    def forward(self, pred,gt,recon1,recon2):

        losses_equal_LF_D = (F.mse_loss(pred, recon2)) * cfg.WEIGHT.EQUAL_R
        losses_equal_LF_L = (F.mse_loss(gt, recon1)) * cfg.WEIGHT.EQUAL_R
        losses_equal_HF_D = (F.mse_loss(pred, recon1)) * cfg.WEIGHT.EQUAL_R
        losses_equal_HF_L = (F.mse_loss(gt, recon2)) * cfg.WEIGHT.EQUAL_R

        enhance_loss = (losses_equal_LF_D + losses_equal_LF_L + losses_equal_HF_D + losses_equal_HF_L)/4

        return enhance_loss

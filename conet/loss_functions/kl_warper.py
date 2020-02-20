import torch.nn
from torch import nn
import torch

class KLWithMask:
    def __call__(self, logits, soft_targets, label_target, ignore_index=-1):
        b, c, w, h = soft_targets.shape
        # soft_targets = torch.clamp(soft_targets, 0, 1)
        # soft_targets = torch.softmax(soft_targets, dim=1)
        # print(soft_targets.max(), soft_targets.min())
        soft_targets = torch.softmax(soft_targets, dim=1)
        soft_targets = soft_targets.clamp(1e-10, 1 - 1e-10)
        
        log_probs = torch.log_softmax(logits, dim=1)

        # soft_loss = nn.functional.kl_div(log_probs, soft_targets, reduction="none")
        # valid_mask = (label_target != ignore_index).long()
        invalid_mask = (label_target == ignore_index).bool()
        invalid_mask = invalid_mask.view(b, 1, w, h).expand(b, c, w, h)

        # print(valid_mask.shape, valid_mask.unique())
        softmax_prob = torch.softmax(logits, dim=1)
        soft_targets[invalid_mask] = softmax_prob[invalid_mask]

        # kl_loss = torch.sum(soft_loss * valid_mask, dim=1).mean()
        # kl_loss = nn.functional.kl_div(log_probs, soft_targets, reduction='batchmean')
        kl_loss = nn.functional.kl_div(log_probs, soft_targets, reduction='batchmean')
        # kl_loss = nn.functional.kl_div(log_probs, soft_targets, reduction='none')
        # kl_loss[invalid_mask] = 0
        # kl_loss = torch.sum(kl_loss, dim=1).mean()
        # kl_loss = kl_loss.mean()

        return kl_loss
        
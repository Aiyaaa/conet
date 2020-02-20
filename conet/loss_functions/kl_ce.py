import torch
import torch.nn.functional as F
from torch import nn

class KLDivergenceCELoss:
    class Config:
        temperature: float = 1.0
        hard_weight: float = 0.0

    def __init__(self, config, ignore_index=-100, weight=None, *args, **kwargs):
        # ignore_index not easily added to kl_div loss, don't support this until needed
        assert ignore_index < 0
        assert 0.0 <= config.hard_weight < 1.0

        self.weight = weight
        self.t = config.temperature
        self.hard_weight = config.hard_weight

    def __call__(self, logits, targets, reduce=True, combine_loss=True):
        """
        Computes Kullback-Leibler divergence loss for multiclass classification
        probability distribution computed by CrossEntropyLoss loss.
        For, KL-divergence, batchmean is the right way to reduce, not just mean.
        """
        hard_targets, _, soft_targets_logits = targets
        soft_targets = F.softmax(soft_targets_logits.float() / self.t, dim=1)
        soft_targets = soft_targets.clamp(1e-10, 1 - 1e-10)
        log_probs = F.log_softmax(logits / self.t, 1)

        if self.weight is not None:
            soft_loss = (
                F.kl_div(log_probs, soft_targets, reduction="none") * self.weight
            )
            # soft_loss dim is batch_size * num_labels, while hard_loss is just
            # batch size, we have to still reduce soft_loss by the labels
            # dimension in order to be able to add the two losses.
            soft_loss = (
                torch.sum(soft_loss, dim=1).mean()
                if reduce
                else torch.sum(soft_loss, dim=1)
            )
        else:
            soft_loss = F.kl_div(
                log_probs, soft_targets, reduction="batchmean" if reduce else "none"
            )

        soft_loss *= self.t ** 2  # See https://arxiv.org/pdf/1503.02531.pdf
        hard_loss = 0.0
        if self.hard_weight > 0.0:
            hard_loss = F.cross_entropy(
                logits,
                hard_targets,
                reduction="mean" if reduce else "none",
                weight=self.weight,
            )

        return (
            (1.0 - self.hard_weight) * soft_loss + self.hard_weight * hard_loss
            if combine_loss
            else (soft_loss, hard_loss)
        )

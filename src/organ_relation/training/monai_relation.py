"""Official MONAI DiceCE on final and native-coarse-logit supervision."""
from torch import nn
from torch.nn import functional as F

from organ_relation.losses import JointLossResult
from .monai_reference import MonaiReferenceLoss, OptimizationLoss, reference_diagnostics
from organ_relation.metrics import hard_dice


def observed_forward(model, image):
    """Opt-in scalar observation of the same forward, never a second model call."""
    if getattr(model, 'relation_scale_enabled', False):
        return model.forward_with_relation_diagnostics(image)
    return model(image), {}


class MonaiRelationLoss(nn.Module):
    def __init__(self, constructor, *, lambda_c, align_corners):
        super().__init__()
        if lambda_c != 0.5 or align_corners is not False:
            raise ValueError('MONAI relation requires lambda_c=0.5 and align_corners=False')
        self.constructor = dict(constructor)
        self.lambda_c, self.align_corners = lambda_c, align_corners
        self.branch = MonaiReferenceLoss(constructor)

    def align_coarse(self, logits, label):
        # Native bottleneck probabilities are used ONLY by the graph. Supervision
        # interpolates logits directly to the original unpadded GT grid.
        return F.interpolate(logits, size=label.shape[1:], mode='trilinear', align_corners=self.align_corners)

    def forward(self, coarse_logits, final_logits, label):
        final = self.branch(final_logits, label)
        coarse = self.branch(self.align_coarse(coarse_logits, label), label)
        total = final.total + self.lambda_c * coarse.total
        return JointLossResult(total, total.reshape(1), coarse.final, final.final)

    def objective(self, coarse_logits, final_logits, label):
        """Keep exactly both official objectives, with no diagnostic allocations.

        Autograd retains what each official loss needs; deleting a Python name
        cannot and must not discard those saved tensors. Coarse is interpolated
        once here and never detached or reconstructed at a different resolution.
        """
        final = self.branch.objective(final_logits, label).total
        coarse = self.branch.objective(self.align_coarse(coarse_logits, label), label).total
        return OptimizationLoss(final + self.lambda_c * coarse, final, coarse)


def relation_diagnostics(output, label, criterion, final_hard):
    result = reference_diagnostics(output.final_logits, label, criterion.branch, final_hard)
    coarse = criterion.align_coarse(output.coarse_logits, label)
    hard = hard_dice(coarse.argmax(1)[0], label[0])
    result['coarse'] = reference_diagnostics(coarse, label, criterion.branch, hard)
    result['coarse']['hard_metrics'] = hard
    result['joint_total_loss'] = result['total_loss'] + criterion.lambda_c * result['coarse']['total_loss']
    return result

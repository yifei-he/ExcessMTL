import torch
import torch.nn.functional as F

from LibMTL.weighting.abstract_weighting import AbsWeighting

class GroupDRO(AbsWeighting):
    def __init__(self):
        super(GroupDRO, self).__init__()
        
    def init_param(self):
        self.loss_weight = torch.tensor([1.0]*self.task_num, device=self.device, requires_grad=False)
        
    def backward(self, losses, **kwargs):

        robust_step_size = kwargs['robust_step_size']
        self.loss_weight = self.loss_weight * torch.exp(losses * robust_step_size)
        self.loss_weight = self.loss_weight / self.loss_weight.sum() * self.task_num
        self.loss_weight = self.loss_weight.detach().clone()
        loss = torch.mul(losses, self.loss_weight).sum()
        loss.backward()

        return self.loss_weight.cpu().numpy()
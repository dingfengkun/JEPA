import torch
import torch.nn.functional as F
from typing import Dict, Tuple
import torch.nn as nn

class JEPALoss:
    """JEPA损失函数的基类"""
    def __call__(self, pred_states, target_states):
        raise NotImplementedError

class VICRegLoss(nn.Module):
    def __init__(
        self,
        inv_coeff: float = 25.0,
        var_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.inv_coeff = inv_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma

    def forward(self, pred_state: torch.Tensor, target_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算VICReg损失
        Args:
            pred_state: 预测的状态表示
            target_state: 目标状态表示
        """
        metrics = dict()
        
        # 不变性损失
        metrics["inv-loss"] = self.inv_coeff * self.representation_loss(pred_state, target_state)
        
        # 方差损失
        metrics["var-loss"] = (
            self.var_coeff
            * (self.variance_loss(pred_state, self.gamma) + self.variance_loss(target_state, self.gamma))
            / 2
        )
        
        # 协方差损失
        metrics["cov-loss"] = (
            self.cov_coeff 
            * (self.covariance_loss(pred_state) + self.covariance_loss(target_state))
            / 2
        )
        
        metrics["loss"] = sum(metrics.values())
        return metrics

    @staticmethod
    def representation_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x, y)

    @staticmethod
    def variance_loss(x: torch.Tensor, gamma: float) -> torch.Tensor:
        x = x - x.mean(dim=0)
        std = x.std(dim=0)
        return F.relu(gamma - std).mean()

    @staticmethod
    def covariance_loss(x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=0)
        cov = (x.T @ x) / (x.shape[0] - 1)
        return cov.fill_diagonal_(0.0).pow(2).sum() / x.shape[1]

class L2Loss(JEPALoss):
    """简单的L2损失"""
    def __call__(self, pred_state, target_state):
        return F.mse_loss(pred_state, target_state)

# 可以在这里添加更多的损失函数类 
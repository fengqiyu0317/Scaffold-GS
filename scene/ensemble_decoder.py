import torch
from torch import nn
import torch.nn.functional as F


class GatedResidualAppearanceDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, residual_scale=0.1, gate_bias=-2.0):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.residual_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters(gate_bias)

    def reset_parameters(self, gate_bias):
        nn.init.zeros_(self.residual_net[-1].weight)
        nn.init.zeros_(self.residual_net[-1].bias)
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, gate_bias)

    def forward(self, x, base_color, return_aux=False):
        residual = torch.tanh(self.residual_net(x))
        gate = torch.sigmoid(self.gate_net(x))
        color = torch.clamp(base_color + self.residual_scale * gate * residual, 0.0, 1.0)
        if not return_aux:
            return color, {}
        aux = {
            "gate_sparse_loss": gate.mean(),
            "gate_mean": gate.detach().mean(),
            "residual_norm_loss": residual.pow(2).mean(),
            "residual_norm": residual.detach().pow(2).mean().sqrt(),
        }
        return color, aux


class TopKRouter(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dim=64, temperature=1.0):
        super().__init__()
        self.num_experts = int(num_experts)
        self.temperature = float(temperature)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, self.num_experts),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, top_k):
        logits = self.net(x) / max(self.temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        k = min(max(1, int(top_k)), self.num_experts)
        top_values, top_indices = torch.topk(logits, k=k, dim=-1)
        top_weights = F.softmax(top_values, dim=-1)
        sparse_weights = torch.zeros_like(logits).scatter(1, top_indices, top_weights)
        return sparse_weights, probs


class MoEAppearanceDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=3, top_k=2, hidden_dim=64, residual_scale=0.1, temperature=1.0):
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.residual_scale = float(residual_scale)
        self.router = TopKRouter(input_dim, self.num_experts, hidden_dim, temperature)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(True),
                nn.Linear(hidden_dim, output_dim),
            )
            for _ in range(self.num_experts)
        ])
        self.reset_parameters()

    def reset_parameters(self):
        for expert in self.experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)

    def forward(self, x, base_color, return_aux=False):
        weights, dense_probs = self.router(x, self.top_k)
        residuals = torch.stack([torch.tanh(expert(x)) for expert in self.experts], dim=1)
        residual = (weights.unsqueeze(-1) * residuals).sum(dim=1)
        color = torch.clamp(base_color + self.residual_scale * residual, 0.0, 1.0)
        if not return_aux:
            return color, {}
        usage = weights.mean(dim=0)
        load_target = usage.new_full((self.num_experts,), 1.0 / self.num_experts)
        entropy = -(dense_probs * torch.log(dense_probs.clamp_min(1e-8))).sum(dim=-1).mean()
        aux = {
            "moe_load_balance_loss": (usage - load_target).pow(2).sum(),
            "router_entropy": entropy.detach(),
            "expert_usage": usage.detach(),
            "residual_norm_loss": residual.pow(2).mean(),
            "residual_norm": residual.detach().pow(2).mean().sqrt(),
        }
        return color, aux

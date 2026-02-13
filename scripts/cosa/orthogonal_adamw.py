import torch
from torch.optim import AdamW

class OrthogonalAdamW(AdamW):
    def __init__(self, *args, beta_ort=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_ort = beta_ort

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                # initialize at first state
                if len(state) == 0:
                    device = p.device
                    state['step'] = torch.tensor(0, dtype=torch.float32, device=device)
                    state['exp_avg'] = torch.zeros_like(p.data, device=device)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, device=device)
                    if group.get('amsgrad', False):
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data, device=device)
                    state['ort_momentum'] = torch.zeros_like(p.data, device=device)

                # orthogonal gradient projection
                ort_momentum = state['ort_momentum']
                dot = torch.sum(grad * ort_momentum)
                norm_sq = torch.sum(ort_momentum * ort_momentum) + group['eps']
                proj = (dot / norm_sq) * ort_momentum if norm_sq > 0 else 0.0
                ort_grad = grad - proj

                # update momentum (c_t in paper)
                ort_momentum.mul_(self.beta_ort).add_(grad, alpha=1 - self.beta_ort)

                # override gradient w/ ort grad
                p.grad.data = ort_grad

        return super().step(closure)

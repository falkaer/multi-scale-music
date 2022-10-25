import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.cuda.amp import autocast

def all_reduce_mean(x):
    dist.all_reduce(x)
    x /= dist.get_world_size()

def noop(*args, **kwargs):
    pass

# CAUTION: multiple forward passes will result in multiple weight updates
class GradNorm(nn.Module):
    def __init__(self, num_weights, shared_param, alpha=0.5, use_ddp=False,
                 optimizer_class=optim.AdamW, **optimizer_kwargs):
        super().__init__()
        self.shared_param = [shared_param]  # wrap to avoid registering
        self.alpha = alpha
        self.all_reduce_fn = all_reduce_mean if use_ddp else noop
        
        self.register_buffer('weights', torch.ones(num_weights, requires_grad=True))
        self.register_buffer('initial_losses', None)
        
        if 'lr' not in optimizer_kwargs:
            optimizer_kwargs['lr'] = 1e-3
        self.optimizer = optimizer_class([self.weights], **optimizer_kwargs)
    
    # TODO: could probably be implemented as an autograd function?
    @autocast(enabled=False)
    def forward(self, task_losses):
        W = self.shared_param[0]
        T = len(self.weights)
        assert len(task_losses) == T
        if self.training:
            all_losses = task_losses.detach().clone()
            gLgW = torch.stack([torch.autograd.grad(L_i, W, retain_graph=True)[0] for L_i in task_losses], dim=0)
            self.all_reduce_fn(all_losses)
            self.all_reduce_fn(gLgW)
            norms = torch.linalg.vector_norm(self.weights[:, None] * gLgW.view(T, -1), dim=-1)
            with torch.no_grad():
                if self.initial_losses is None:
                    self.initial_losses = all_losses
                loss_ratios = all_losses / self.initial_losses
                inverse_train_rates = loss_ratios / loss_ratios.mean()
                constant_term = norms.mean() * inverse_train_rates ** self.alpha
            grad_norm_loss = (norms - constant_term).abs().sum()
            self.weights.grad = torch.autograd.grad(grad_norm_loss, self.weights)[0]
            self.optimizer.step()
            with torch.no_grad():
                self.weights.data = self.weights / self.weights.sum() * T
        return self.weights.detach() * task_losses
    
    def get_extra_state(self):
        return self.optimizer.state_dict()
    
    def set_extra_state(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

if __name__ == '__main__':
    torch.manual_seed(0)
    N = nn.Linear(10, 5)
    q = nn.Parameter(torch.tensor([10., 1.]))
    
    G = GradNorm(2, N.weight, alpha=0.5, lr=1e-2)
    print(list(G.parameters()))
    
    x = torch.randn(2, 10)
    y = N(x)
    Ls = q * y.mean(dim=1)
    print(Ls)
    
    for i in range(100):
        Ls_normed = G(Ls)
    
    print(G.weights)
    print(G.state_dict())

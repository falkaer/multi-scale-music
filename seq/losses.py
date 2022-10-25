import torch
import torch.nn.functional as F

def cross_entropy(logits, target):
    C = logits.shape[-1]
    assert logits.shape[:3] == target.shape[:3]
    if logits.ndim == target.ndim:
        target = target.view(-1, C)
    else:
        target = target.view(-1)
    logits = logits.view(-1, C)
    return F.cross_entropy(logits, target, reduction='mean')

def weighted_cross_entropy(logits, target, weights, dim=2):
    loss = 0.
    assert weights.ndim == 1
    assert logits.shape[dim] == target.shape[dim] == weights.shape[0]
    for w, l, t in zip(weights, logits.unbind(dim), target.unbind(dim)):
        loss = loss + w * cross_entropy(l, t)
    return loss / len(weights)

if __name__ == '__main__':
    torch.manual_seed(0)

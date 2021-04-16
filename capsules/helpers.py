import torch


def squash(x,
           squashing_constant=1.,
           dim=-1,
           eps=1e-7,
           safe=True,
           p=2,
           **kwargs):
    if safe:
        squared_norm = torch.sum(torch.square(x), axis=dim, keepdim=True)
        safe_norm = torch.sqrt(squared_norm + eps)
        squash_factor = squared_norm / (squashing_constant + squared_norm)
        unit_vector = x / safe_norm

        return squash_factor * unit_vector
    else:
        norm = x.norm(dim=dim, keepdim=True, p=p)
        norm_squared = norm * norm
        return (x / norm) * (norm_squared / (squashing_constant + norm_squared))


def smsquash(x, caps_dim=1, atoms_dim=2, eps=1e-7, **kwargs):
    """Softmax Squash (Poiret, et al., 2021)
    Novel squash function. It rescales caps 2-norms to a probability
    distribution over predicted output classes."""
    squared_norm = torch.sum(torch.square(x), axis=atoms_dim, keepdim=True)
    safe_norm = torch.sqrt(squared_norm + eps)

    a = torch.exp(safe_norm) / torch.sum(torch.exp(safe_norm), axis=caps_dim)
    b = x / safe_norm

    return a * b

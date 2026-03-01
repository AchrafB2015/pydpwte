import torch

from pydpwte.dpwte.dpwte import dpwte
from pydpwte.utils.loss import total_loss


def test_dpwte_forward_and_loss():
    n_cols = 5
    p_max = 3

    model = dpwte(n_cols=n_cols, p_max=p_max, sparse_reg=False)

    x = torch.rand(10, n_cols)

    times = torch.rand(10)
    deltas = torch.randint(low=0, high=2, size=(10,), dtype=torch.float32)
    y = torch.stack([times, deltas], dim=1)

    loss = total_loss(model, x, y)

    assert loss is not None


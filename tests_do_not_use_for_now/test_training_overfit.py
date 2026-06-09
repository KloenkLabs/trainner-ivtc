import torch
from torch import nn

from trainner_ivtc.grid import GRID_IGNORE_INDEX, dense_targets_from_labels
from trainner_ivtc.model import GlobalCadenceClassifier


def test_uniform_label_dense_loss_accepts_current_batches() -> None:
    torch.manual_seed(1)
    model = GlobalCadenceClassifier(in_channels=22, base_channels=4, channel_mult=(1, 2), dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=GRID_IGNORE_INDEX)
    fields = torch.cat([torch.zeros(4, 22, 16, 24), torch.ones(4, 22, 16, 24)], dim=0)
    labels = torch.tensor([0, 0, 0, 0, 5, 5, 5, 5])
    first_loss = None
    last_loss = None
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        logits = model(fields)
        targets = dense_targets_from_labels(labels, int(logits.shape[-2]), int(logits.shape[-1]))
        loss = criterion(logits, targets)
        if first_loss is None:
            first_loss = float(loss.detach().item())
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().item())
    assert first_loss is not None
    assert last_loss is not None
    assert last_loss < first_loss

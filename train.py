import typing as t

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from myutils import DEVICE, synthesize_data, _corners
from detector_fpn import Detector_FPN


def compute_loss(pred, target):
    """Compute loss handling no ships
    Arguments:
        pred {Tensor Batch} -- p(ship), x, y, yaw, w, h
        target {Tensor Batch} -- p(ship), x, y, yaw, w, h
    Returns:
        loss -- list of all - not averaged
    """
    assert pred.shape[-1] == 6
    assert target.shape[-1] == 6

    # instances with no ships to pred bbox
    idx_no_ship = torch.nonzero(target[:, 0] == 0, as_tuple=True)

    l_bbox = lmr5p(pred[:, 1:], target[:, 1:])
    l_bbox[idx_no_ship] = 0

    l_ship = torch.nn.functional.binary_cross_entropy_with_logits(
        pred[:, 0], target[:, 0], reduction="none"
    )

    loss = l_ship + l_bbox

    return loss

def lmr5p(pred, target):
    """5 parameter modulated rotation loss
    Arguments:
        pred {Tensor Batch} -- x, y, yaw, w, h
        target {Tensor Batch} -- x, y, yaw, w, h
        * X and Y position (centre of the bounding box)
        * Yaw (direction of heading)
        * Width (size tangential to the direction of yaw)
        * Height (size along the direct of yaw)
    Returns:
        loss for each pred, target pair without sum
    Reference: Eqn(2) https://arxiv.org/pdf/1911.08299.pdf
    """
    assert pred.shape[-1] == 5
    assert target.shape[-1] == 5

    x1, x2 = pred[:, 0], target[:, 0]
    y1, y2 = pred[:, 1], target[:, 1]
    yaw1, yaw2 = pred[:, 2], target[:, 2]
    w1, w2 = pred[:, 3], target[:, 3]
    h1, h2 = pred[:, 4], target[:, 4]

    # center point loss
    lcp = torch.abs(x1 - x2) + torch.abs(y1 - y2)

    lmr5p_ = torch.min(
        lcp + torch.abs(w1 - w2) + torch.abs(h1 - h2) + torch.abs(yaw1 - yaw2),
        lcp
        + torch.abs(w1 - h2)
        + torch.abs(h1 - w2)
        + torch.abs(90 - torch.abs(yaw1 - yaw2)),
    )

    return lmr5p_

class StarDataset(torch.utils.data.Dataset):
    """Return star image and labels"""

    def __init__(self, data_size=50000):
        self.data_size = data_size

    def __len__(self) -> int:
        return self.data_size

    def __getitem__(self, idx) -> t.Tuple[torch.Tensor, torch.Tensor]:
        image, label = synthesize_data(has_star=True)
        if np.isnan(label[0]):
            label = np.insert(label, 0, 0, axis=0)
        else:
            label[2] *= 57.2958
            label = np.insert(label, 0, 1, axis=0)
        return image[None], label


def train(model, dl: StarDataset, num_epochs: int):

    loss_fn = compute_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        print(f"EPOCH: {epoch}")
        losses = []
        for image, label in tqdm(dl, total=len(dl)):
            image = image.to(DEVICE).float()
            label = label.to(DEVICE).float()
            optimizer.zero_grad()
            preds = model(image)
            loss = loss_fn(preds,label)
            loss = loss.mean()
            loss.backward()
            losses.append(loss.detach().cpu().numpy())
            optimizer.step()

    return model


model = Detector_FPN().to(DEVICE)

star_model = train(
    model,
    torch.utils.data.DataLoader(StarDataset(), batch_size=64, num_workers=8),
    num_epochs=20,
)

torch.save(star_model.state_dict(), "model.pickle")
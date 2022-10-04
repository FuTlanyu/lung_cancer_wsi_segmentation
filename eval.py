import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import new_dice_coeff


def eval_net(net, loader, device, n_val):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot1 = 0
    tot2 = 0

    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            true_masks = batch['mask']

            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)

            mask_pred = net(imgs)

            for true_mask, pred in zip(true_masks, mask_pred):
                pred = (torch.sigmoid(pred) > 0.5).float()
                if net.n_classes > 1:
                    tot1 += F.cross_entropy(pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0)).item()
                else:
                    # tot += dice_coeff(pred, true_mask.squeeze(dim=1)).item()
                    tot1 += new_dice_coeff(pred, true_mask.squeeze(dim=1)).item()[0]
                    tot2 += new_dice_coeff(pred, true_mask.squeeze(dim=1)).item()[1]
            pbar.update(imgs.shape[0])

    return tot1 / tot2

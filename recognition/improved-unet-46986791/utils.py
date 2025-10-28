import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, u, v):
        # From the paper:
        # - "u is the softmax output of the network"
        # - "v is a one hot encoding of the ground truth segmentation map"
        # the equation is basically summed intersection over union multiplied by -2/(num classes)

        # flatten
        u_flat = u.flatten(start_dim=2)
        v_flat = v.flatten(start_dim=2)
        
        # Numerator sigmas
        intersection = (u_flat * v_flat).sum(dim=2)
        union = u_flat.sum(dim=2) + v_flat.sum(dim=2)

        # Evaluate fraction
        dice_per_class = intersection / (union + 1e-8)
        
        # Big sigma in front of fraction
        dice_sum = dice_per_class.sum(dim=1)

        # Modifier out front of loss equation (6 classes, -2 constant from paper)
        loss = (-2)/6 * dice_sum.mean(dim=0)

        return loss
    

def per_class_score_dice(u, v):
    classes = u.shape[1]
    
    u = torch.argmax(u, dim=1)
    v = torch.argmax(v, dim=1)
    
    dice_scores = torch.zeros(classes)
    for k in range(classes):
        u_k = (u == k).float()
        v_k = (v == k).float()

        # Don't sum over batch, only over h and w.
        intersection = (u_k * v_k).sum(dim=(1, 2))
        union = u_k.sum(dim=(1, 2)) + v_k.sum(dim=(1, 2))

        dice_per_image = (2.0 * intersection) / (union + 1e-8)

        # If an image doesn't have a class, we don't want to incorrectly assign a 0 score
        # as this will drag the average down, even though the model is not incorrect
        # Rather, we should just not include it in the average
        gt_has_class = v_k.sum(dim=(1, 2)) > 0
        
        if gt_has_class.any():
            dice_scores[k] = dice_per_image[gt_has_class].mean()
        else:
            # edge case for when a class doesn't appear in a batch
            # rare but possible since we shuffle our batches)
            dice_scores[k] = float('nan')

    return dice_scores
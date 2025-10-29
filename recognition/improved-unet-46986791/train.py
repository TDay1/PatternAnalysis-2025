import torch
from torch.utils.data import DataLoader
from dataset import HipMRIDataset
from modules import ImprovedUNet
from utils import DiceLoss, per_class_score_dice, build_transforms
from tqdm.auto import tqdm
import os
import pandas as pd
import argparse
from torch.optim.lr_scheduler import ExponentialLR

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", default=20, type=int)
parser.add_argument("-lr", "--learning-rate", default=1e-3, type=float)
parser.add_argument("-lrd", "--learning-rate-decay-gamma", default=0.985, type=float)
parser.add_argument("-bs", "--batch-size", default=32, type=int)
parser.add_argument("-o", "--output-dir", default="./checkpoints", type=str)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu", type=str)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# data
transforms = build_transforms()
train_ds = HipMRIDataset('./data/keras_slices_data/keras_slices_train', './data/keras_slices_data/keras_slices_seg_train', transforms=transforms)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_worker=12, pin_memory=(args.device == "cuda"))

val_ds = HipMRIDataset('./data/keras_slices_data/keras_slices_validate', './data/keras_slices_data/keras_slices_seg_validate')
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,  num_worker=4, pin_memory=(args.device == "cuda"))

model = ImprovedUNet().to(args.device)
loss_fn = DiceLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
scheduler = ExponentialLR(optimiser, gamma=args.learning_rate_decay_gamma)

# Logging
train_csv_dir = f'./{args.output_dir}/train_metrics.csv'
train_metrics = []
val_csv_dir = f'./{args.output_dir}/val_metrics.csv'
val_metrics = []


for epoch in range(args.epochs):
    print(f"======== Epoch {epoch}/{args.epochs} ======")

    print(f"Training...")
    loading_bar = tqdm(train_loader)

    train_loss = 0.0
    loss_history = []
    dice_class_scores = []
    
    model.train()
    for batch_index, (images, segs) in enumerate(loading_bar):
        images = images.to(args.device)
        segs = segs.to(args.device)

        optimiser.zero_grad()
        outputs = model(images)

        loss = loss_fn(outputs, segs)
        loss.backward()

        optimiser.step()

        train_loss += loss.item()
        loss_history.append(loss.item())

        class_dice = per_class_score_dice(outputs, segs)
        dice_class_scores.append(class_dice)

        loading_bar.set_postfix({"Total loss": f"{train_loss:.4f}", "mean loss": f"{(train_loss/len(loss_history)):.4f}", "Current LR": scheduler.get_last_lr()[0]})

        # log batch
        train_metrics.append({
            'epoch': epoch,
            'batch': batch_index,
            'batch_loss': loss.item(),
            'class_0_dice': class_dice[0].item(),
            'class_1_dice': class_dice[1].item(),
            'class_2_dice': class_dice[2].item(),
            'class_3_dice': class_dice[3].item(),
            'class_4_dice': class_dice[4].item(),
            'class_5_dice': class_dice[5].item(),
            'batch_mean_dice': class_dice.mean().item(),
            'current_lr': scheduler.get_last_lr()[0],
        })

    pd.DataFrame(train_metrics).to_csv(train_csv_dir)

    dice_class_scores = torch.stack(dice_class_scores)
    mean_dice_class_scores = torch.nanmean(dice_class_scores, dim=0)
    
    print(f"class-by-class mean dice score on train set: {mean_dice_class_scores}")


    # Validation
    print(f"Validating...")
    loading_bar = tqdm(val_loader)
    model.eval()

    val_loss = 0
    loss_history = []
    dice_class_scores = []
    
    with torch.no_grad():
        for images, segs in loading_bar:
            images = images.to(args.device)
            segs = segs.to(args.device)

            outputs = model(images)

            loss = loss_fn(outputs, segs)
            val_loss += loss.item()
            loss_history.append(loss.item())

            dice_class_scores.append(per_class_score_dice(outputs, segs))
            loading_bar.set_postfix({"Total loss": f"{val_loss:.4f}", "mean loss": f"{(val_loss/len(loss_history)):.4f}"})



    dice_class_scores = torch.stack(dice_class_scores)
    mean_dice_class_scores = torch.nanmean(dice_class_scores, dim=0)
    
    print(f"class-by-class mean dice score on validation set: {mean_dice_class_scores}")

    # log val epoch
    val_metrics.append({
        'epoch': epoch,
        'mean_loss': (val_loss/len(loss_history)),
        'class_0_dice': mean_dice_class_scores[0].item(),
        'class_1_dice': mean_dice_class_scores[1].item(),
        'class_2_dice': mean_dice_class_scores[2].item(),
        'class_3_dice': mean_dice_class_scores[3].item(),
        'class_4_dice': mean_dice_class_scores[4].item(),
        'class_5_dice': mean_dice_class_scores[5].item(),
        'epoch_mean_dice': mean_dice_class_scores.mean().item(),
    })

    pd.DataFrame(val_metrics).to_csv(val_csv_dir)

    torch.save(model.state_dict(), f'{args.output_dir}/epoch_{epoch}.pth')

    scheduler.step()

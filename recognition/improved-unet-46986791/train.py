import torch
from torch.utils.data import DataLoader
from dataset import HipMRIDataset
from modules import ImprovedUNet
from utils import DiceLoss, per_class_score_dice
from tqdm.auto import tqdm

device = 'mps'

# data
train_ds = HipMRIDataset('./data/keras_slices_data/keras_slices_train', './data/keras_slices_data/keras_slices_seg_train')
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

model = ImprovedUNet().to(device)
loss_fn = DiceLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10
for epoch in range(num_epochs):
    train_loss = 0.0
    loss_history = []
    dice_class_scores = []
    
    loading_bar = tqdm(train_loader)
    model.train()
    print(f"Training epoch {epoch}/{num_epochs}")
    for images, segs in loading_bar:
        images = images.to(device)
        segs = segs.to(device)

        optimiser.zero_grad()
        outputs = model(images)

        loss = loss_fn(outputs, segs)
        loss.backward()

        optimiser.step()

        train_loss += loss.item()
        loss_history.append(loss.item())

        dice_class_scores.append(per_class_score_dice(outputs, segs))

        loading_bar.set_postfix({"Total loss": f"{train_loss:.4f}", "mean loss": f"{(train_loss/len(loss_history)):.4f}"})

    dice_class_scores = torch.stack(dice_class_scores)
    mean_dice_class_scores = torch.nanmean(dice_class_scores, dim=0)
    
    print(f"class-by-class mean dice score: {mean_dice_class_scores}")

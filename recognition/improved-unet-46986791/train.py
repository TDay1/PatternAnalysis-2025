import torch
from torch.utils.data import DataLoader
from dataset import HipMRIDataset
from modules import ImprovedUNet
from utils import DiceLoss
from tqdm.auto import tqdm

device = 'mps'

# data
train_ds = HipMRIDataset('./data/keras_slices_data/keras_slices_train', './data/keras_slices_data/keras_slices_seg_train')
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

model = ImprovedUNet().to(device)
loss_fn = DiceLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 20
model.train()
for epoch in range(num_epochs):
    train_loss = 0.0
    loss_history = []

    loading_bar = tqdm(train_loader)
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

        loading_bar.set_postfix({"Total loss": f"{train_loss:.4f}", "mean loss": f"{(train_loss/len(loss_history)):.4f}"})

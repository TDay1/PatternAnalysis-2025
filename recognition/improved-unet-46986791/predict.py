from modules import ImprovedUNet
import torch
from dataset import HipMRIDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import os

# Parse CLI args
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model-dir", required=True, type=str)
parser.add_argument("-o", "--output-dir", default="./predict", type=str)
parser.add_argument("-c", "--count", default=2, type=int)
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

# For such a low number of examples, using the CPU is fine and simplifies things
device = "cpu"

# Insanciate model + load weights
model = ImprovedUNet()
state_dict = torch.load(args.model_dir, map_location=torch.device(device))
model.load_state_dict(state_dict)
model.eval()

# Load dataset
test_ds = HipMRIDataset('./data/keras_slices_data/keras_slices_test', './data/keras_slices_data/keras_slices_seg_test')
test_loader = DataLoader(test_ds, batch_size=args.count, shuffle=True)
images, segs = next(iter(test_loader))

# Predict
outputs = model(images)

# Convert the predicitons and labels to label-encoding
predictions = torch.argmax(outputs, dim=1)
segs_label = torch.argmax(segs, dim=1)

# plot raw segmentations next to input image
fig, axes = plt.subplots(args.count, 3)
for i in range(args.count):
    axes[i, 0].imshow(images[i, 0], cmap='gray')
    axes[i, 0].set_title("Input MRI")
    axes[i, 0].axis('off')

    axes[i, 1].imshow(segs_label[i], vmin=0, vmax=5)
    axes[i, 1].set_title("Ground Truth\nSegmentation")
    axes[i, 1].axis('off')

    axes[i, 2].imshow(predictions[i], vmin=0, vmax=5)
    axes[i, 2].set_title("Predicted\nSegmentation")
    axes[i, 2].axis('off')

fig.tight_layout()
fig.savefig(f"{args.output_dir}/segmentation_pred.png")

# Overlay segmentations on input image
fig, axes = plt.subplots(args.count, 2)
for i in range(args.count):
    axes[i, 0].imshow(images[i, 0], cmap='gray')
    axes[i, 0].imshow(segs_label[i], alpha=0.5, cmap="tab10", vmin=0, vmax=5)
    axes[i, 0].set_title("Ground Truth")
    axes[i, 0].axis('off')

    axes[i, 1].imshow(images[i, 0], cmap='gray')
    axes[i, 1].imshow(predictions[i], alpha=0.5, cmap="tab10", vmin=0, vmax=5)

    axes[i, 1].set_title("Predicted")
    axes[i, 1].axis('off')

fig.tight_layout()
fig.savefig(f"{args.output_dir}/overlay_pred.png")

from modules import ImprovedUNet
import torch
from dataset import HipMRIDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = "cpu"
examples = 2

model = ImprovedUNet().to(device)
state_dict = torch.load('./checkpoint/epoch_35.pth', map_location=torch.device(device))
model.load_state_dict(state_dict)
model.eval()

test_ds = HipMRIDataset('./data/keras_slices_data/keras_slices_test', './data/keras_slices_data/keras_slices_seg_test')
test_loader = DataLoader(test_ds, batch_size=examples, shuffle=True)


images, segs = next(iter(test_loader))
images = images.to(device)
segs = segs.to(device)

outputs = model(images)
predictions = torch.argmax(outputs, dim=1)
segs_label = torch.argmax(segs, dim=1)

print(images.shape)
print(predictions.shape)

# plot
fig, axes = plt.subplots(examples, 3)
for i in range(examples):
    axes[i, 0].imshow(images[i, 0], cmap='gray')
    axes[i, 0].set_title("Input MRI")
    axes[i, 0].axis('off')

    axes[i, 1].imshow(segs_label[i], vmin=0, vmax=5)
    axes[i, 1].set_title("Ground Truth Segmentation")
    axes[i, 1].axis('off')

    axes[i, 2].imshow(predictions[i])
    axes[i, 2].set_title("Predicted Segmentation")
    axes[i, 2].axis('off')

fig, axes = plt.subplots(examples, 2)
for i in range(examples):
    axes[i, 0].imshow(images[i, 0], cmap='gray')
    axes[i, 0].imshow(segs_label[i], alpha=0.5, cmap="tab10", vmin=0, vmax=5)
    axes[i, 0].set_title("Ground Truth")
    axes[i, 0].axis('off')

    axes[i, 1].imshow(images[i, 0], cmap='gray')
    axes[i, 1].imshow(predictions[i], alpha=0.5, cmap="tab10", vmin=0, vmax=5)

    axes[i, 1].set_title("Predicted")
    axes[i, 1].axis('off')

plt.tight_layout()
plt.show()
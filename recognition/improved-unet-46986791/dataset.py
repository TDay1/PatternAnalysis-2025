import numpy as np
import nibabel as nib
import os
import glob
from torch.utils.data import Dataset
from scipy.ndimage import zoom
import torch
from torchvision import tv_tensors

# The below two functions (to_channels and load_data_2D) are based on the
# sample code provided in the assignment task sheet (appendix B)
# The to_channels function was modified such that a the number of channels
# could be manually specified, which avoids a bug when some segmentations
# contain a lesser number of segmentations than others.

def to_channels(arr: np.ndarray, num_channels: int = 6, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (num_channels,), dtype=dtype)
    
    for c in channels:
        c = int(c)
        res[..., c][arr == c] = 1
    
    return res

# load medical image functions
def load_data_2D(imageNames, normImage=False, categorical=False , dtype=np.float32, getAffines=False, early_stop=False):
    '''
    Load medical image data from names , cases list provided into a list for each.
    This function pre - allocates 4 D arrays for conv2d to avoid excessive memory
    usage.
    
    normImage : bool ( normalise the image 0.0 -1.0)
    early_stop : Stop loading pre-maturely , leaves arrays mostly empty, for quick
    loading and testing scripts.
    '''
    affines = []
    
    # get fixed size
    num = len (imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len (first_case.shape) == 3:
        first_case = first_case [: ,: ,0] # sometimes extra dims , remove
    if categorical:
        first_case = to_channels (first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, 6), dtype=dtype)
    else :
        rows, cols = first_case.shape
        images = np.zeros((num ,rows, cols), dtype=dtype)
    
    for i, inName in enumerate(imageNames):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged') # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage [:,:,0] # sometimes extra dims in HipMRI_study data
        
        inImage = inImage.astype(dtype)
        if normImage :
            # ~ inImage = inImage / np . linalg . norm ( inImage )
            # ~ inImage = 255. * inImage / inImage . max ()
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical :
            inImage = to_channels(inImage, dtype=dtype)
            images[i,:,:,:] = inImage
        else :
            images[i,:,:] = inImage
        
        affines.append(affine)
        
        if i > 20 and early_stop:
            break
    
    if getAffines:
        return images , affines
    else:
        return images
    

class HipMRIDataset(Dataset):
    def __init__(self, img_dir, seg_dir, transforms=None):
        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
        self.seg_files = sorted(glob.glob(os.path.join(seg_dir, "*.nii.gz")))
        self.transforms = transforms

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = load_data_2D([self.img_files[idx]] , normImage = True , categorical = False)
        seg = load_data_2D([self.seg_files[idx]], normImage = False , categorical = True)

        # if our image isn't (256, 128), resize it.
        if img.shape[1:] != (256, 128):
            img_zoom_ratios = (1, 256/img.shape[1], 128 / img.shape[2])
            img = zoom(img, img_zoom_ratios, order=1)

            seg_zoom_ratios = (1, 256/seg.shape[1], 128 / seg.shape[2], 1)
            seg = zoom(seg, seg_zoom_ratios, order=0)

        x = img[0, None, :, :]
        y = np.transpose(seg[0], (2, 0, 1))

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        x = tv_tensors.Image(x)
        y = tv_tensors.Mask(y)

        if self.transforms is not None:
            x, y = self.transforms(x, y)
        
        return x, y
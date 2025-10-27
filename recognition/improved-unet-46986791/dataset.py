import numpy as np
import nibabel as nib
from tqdm import tqdm
import os
import glob
from torch.utils.data import Dataset

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
    
    for i, inName in enumerate(tqdm(imageNames)):
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
    def __init__(self, img_dir, seg_dir):
        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
        self.seg_files = sorted(glob.glob(os.path.join(seg_dir, "*.nii.gz")))

        self.images = load_data_2D(self.img_files , normImage = True , categorical = False)
        self.segs = load_data_2D(self.seg_files , normImage = False , categorical = True)

        # Reshape images and segmentations to be in expected shape
        self.images = self.images[:, None, :, :]
        self.segs = np.transpose(self.segs, (0, 3, 1, 2))

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.segs[idx]
        
        return x, y
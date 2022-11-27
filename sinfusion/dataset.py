from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomCrop, Resize, ToTensor

from PIL import Image


class CropImageDataset(Dataset):
    def __init__(self, image, resolution=128):
        """image -- torch.tensor or path to image file"""
        super(Dataset, self).__init__()
        if type(image) == str:
            image = Image.open(image)
            image = ToTensor()(image)
        self.image = image
        h, w = image.shape[-2:]
        self.t = Compose([
            RandomCrop(int(0.95 * min(h, w))),
            # Resize(resolution)
        ])

    def __len__(self):
        return 128

    def __getitem__(self, idx):
        return self.t(self.image)

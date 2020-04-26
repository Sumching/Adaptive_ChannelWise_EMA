import os.path
import torch
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import random
T = transforms


class DataMyload(Dataset):

    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.transform = transform
        self.trainA = glob(os.path.join(image_dir, 'trainA', '*.jpg'))
        random.shuffle(self.trainA)
        print('images: ', len(self.trainA))
        self.trainB = glob(os.path.join(image_dir, 'trainB', '*.jpg'))
        random.shuffle(self.trainB)
        self.A_size = len(self.trainA)
        self.B_size = len(self.trainB)

    def __getitem__(self, index):
        filenameA = self.trainA[index % self.A_size]
        filenameB = self.trainB[index % self.B_size] 
        imageA = Image.open(filenameA).convert('RGB')
        imageB = Image.open(filenameB).convert('RGB')
        return self.transform(imageA), self.transform(imageB)

    def __len__(self):
        """Return the number of images."""
        return len(self.trainA)


def get_loader(image_dir='../data/edges2shoes', batch_size=16, image_size=256, num_workers=20):
    """Build and return a data loader."""
    transform = []
    transform.append(T.RandomHorizontalFlip())
    #transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = DataMyload(image_dir, transform)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=num_workers)
    return data_loader
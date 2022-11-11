import torch
import cv2
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class ImageDataset(Dataset):

    def __init__(self, image_dir, pattern=".jpg"):
        """Initialize the attributes of the object of the class."""
        self.image_dir = image_dir
        self.image_path_list = sorted(self.find_files(image_dir, pattern))

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.image_path_list)

    def __getitem__(self, index):
        """Return a data sample (=image) for a given index, along with the name of the corresponding pokemon."""
        
        image_path = self.image_path_list[index]
        return self.image_path_list[index]
        #img = cv2.imread(image_path)  # numpy array
        #print(f"tito was here {img.shape}")
        #return img
        


    def find_files(self, directory, pattern):

        return  [f.path for f in os.scandir(directory) if f.path.endswith(pattern)]  # ends with does not like regex

def visual_collate_fn(batch):
    # TODO: Implement your function
    # https://python.plainenglish.io/understanding-collate-fn-in-pytorch-f9d1742647d3
    # But I guess in your case it should be:
    raw_images = [cv2.imread(image) for image in batch]
    heights, widths = zip(*[im.shape[:2] for im in raw_images])
    up_height = max(heights)
    up_width = max(widths)
    up_points = (up_width, up_height)

    resized_images = [cv2.resize(image, up_points, interpolation= cv2.INTER_LINEAR) for image in raw_images]

    #print([i.shape for i in raw_images])
    #print([i.shape for i in resized_images])
    return [torch.tensor(im) for im in resized_images] #(3)  # torch.from_numpy

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
        


    def find_files(self, directory, pattern):

        return  [f.path for f in os.scandir(directory) if f.path.endswith(pattern)]  # ends with does not like regex

class Rescale:

    def __init__(self, heights, widths, scaler_method=min):
        height = scaler_method(heights)
        width = scaler_method(widths)
        self.points = (width, height)

    def up_down_scaler(self, image):
        """
        scalar method dependent on init
        """
        return cv2.resize(image, self.points, interpolation=cv2.INTER_LINEAR)


def visual_collate_fn(batch, scaler_method):
    # TODO: Implement your function
    # https://python.plainenglish.io/understanding-collate-fn-in-pytorch-f9d1742647d3
    # https://learnopencv.com/image-resizing-with-opencv/
    # But I guess in your case it should be:
    raw_images = [cv2.imread(image) for image in batch]
    heights, widths = zip(*[im.shape[:2] for im in raw_images])
    rescaler = Rescale(heights, widths, scaler_method=scaler_method)

    #image_storer(raw_images, False)  # only uncomment if you want to store images.

    #image = cv2.imread(batch[0])
    #cv2.imshow('rar', image)
    #cv2.waitKey(5000) 
    #cv2.destroyAllWindows()
    #cv2.imwrite('og_img.jpg', raw_images[0])

    #https://stackoverflow.com/questions/58100252/jupyter-kernel-crashes-when-trying-to-display-image-with-opencv

    resized_images = [rescaler.up_down_scaler(image) for image in raw_images]

    #image_storer(resized_images, True)  # only uncomment if you want to store images.

    #print([i.shape for i in raw_images])
    #print([i.shape for i in resized_images])
    return [torch.tensor(im) for im in resized_images] #(3)  # torch.from_numpy

# THIS FUNCTION IS ONLY HERE TO PRINT IMAGES FOR PRESENTATION PURPOSES
def image_storer(im_batch, resized):
    if resized:
        im_name = "resized_demo_image_"
    else:
        im_name = "original_demo_image_"


    for i, im in enumerate(im_batch):
        cv2.imwrite(im_name + str(i) + ".png", im)

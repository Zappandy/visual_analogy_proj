import torch
import cv2
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class ResizeImageDataset(Dataset):

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

class CutterImageDataset(ResizeImageDataset):

    def __init__(self):
        super.__init__()
        self.cutter = ImageCutter
        self.create_cut_batch(self.image_path_list)

    def create_cut_batch(self, index):
        for path in self.image_path_list:

            path = self.image_path_list[index]
            im = cv2.imread(file_path)
            height, width = im.shape[:2]
            image_cut_vals = self.cutter(width, height)
            corr_width = image_cut_vals.corr_width
            corr_height = image_cut_vals.corr_height
        self.tile_generator()

    def tile_generator(self, width=width, height=height, corr_width, corr_height=corr_height):
        all_tiles = list()
        for y in range(0, height, corr_height):  # 800, 200
            for x in range(0, width, corr_width):  # 526, 263
                y1 = y + corr_height
                x1 = x + corr_width
                tiles = image_copy[y:y+corr_height, x:x+corr_width]
                # for viz comment for now
                #cv2.imwrite('tile'+str(y1)+'_'+str(x1)+'.jpg', tiles)
                #cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 1)
                all_tiles.append(tiles)

        # for viz comment for now
        #plt.imshow(image[:, :, ::-1])
        #plt.axis('off')
        #cv2.imwrite("patched.jpg",image)

        return all_tiles

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

class ImageCutter:

    def __init__(self, width, height):
        self.corr_width = self.crop_values(width)
        self.corr_height = self.crop_values(height)

    def crop_values(self, measure):
        divisors = [n for n in range(2, 10) if measure%n == 0]  # we don't want insane amount of crops
        if len(divisors) == 1:
            result = divisors[0]    
        elif not divisors:
            result = measure
        else:
            if len(divisors) % 2 != 0:
                result = int(np.median(divisors))
            else:
                idx = len(divisors) // 2
                result = divisors[idx-1]  # upper end == idx    
        return measure // result


def visual_collate_fn(batch, scaler_method, naive_approach=False):
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
    return [torch.tensor(im) for im in resized_images]

# THIS FUNCTION IS ONLY HERE TO PRINT IMAGES FOR PRESENTATION PURPOSES
def image_storer(im_batch, resized):
    if resized:
        im_name = "resized_demo_image_"
    else:
        im_name = "original_demo_image_"


    for i, im in enumerate(im_batch):
        cv2.imwrite(im_name + str(i) + ".png", im)

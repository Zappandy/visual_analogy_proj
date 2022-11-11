import torch
import cv2
import os
from torch.utils.data import Dataset, DataLoader
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from data_config import *

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
        img = cv2.imread(image_path)
        # dims are either x * 500 or 500 * x
        # [3, 330, 500], [3, 650, 200]  #  view(-1)
        #name = image_path.replace(self.image_dir, '').replace('.png', '')
        #x = io.imread(image_path)
        #x = torch.tensor(x, dtype=float)
        
        #return x, name
        x = torch.tensor(img, dtype=float)
        return x
        #return img


    def find_files(self, directory, pattern):

        return  [f.path for f in os.scandir(directory) if f.path.endswith(pattern)]  # ends with does not like regex


visual_genome_pttrn = ".jpg"
# images_1 https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
# https://medium.com/analytics-vidhya/how-to-load-any-image-dataset-in-python-3bd2fa2cb43d
# https://stackoverflow.com/questions/65279115/how-to-use-collate-fn-with-dataloaders
# https://medium.com/analytics-vidhya/how-to-load-any-image-dataset-in-python-3bd2fa2cb43d
visual_genome = ImageDataset(VG_PATH, pattern=visual_genome_pttrn)
flickr30k = ImageDataset(FLICKR_PATH, pattern=visual_genome_pttrn)
#print(visual_genome[:5])  # this slicing does not work, need to be loaded to data_loader 
vg_dataloader = DataLoader(visual_genome, batch_size=BATCH_SIZE, shuffle=False)
flickr_dataloader = DataLoader(flickr30k, batch_size=BATCH_SIZE, shuffle=False)
#print(visual_genome[:5])

#4 cap 4 images --> analogies on imgs and captions

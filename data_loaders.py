import torch
import os
from torch.utils.data import Dataset

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
        #name = image_path.replace(self.image_dir, '').replace('.png', '')
        #x = io.imread(image_path)
        #x = torch.tensor(x, dtype=float)
        
        #return x, name
        return image_path


    def find_files(self, directory, pattern):

        return  [f.path for f in os.scandir(directory) if f.path.endswith(pattern)]  # ends with does not like regex


base_path = "/media/andres/2D2DA2454B8413B5/software_proj"
visual_genome_pttrn = ".jpg"
visual_genome_path = "/visual_genome/VG_100K/"

visual_path = base_path + visual_genome_path
visual_dataset = ImageDataset(visual_path)
print(visual_dataset[:5])

#4 cap 4 images --> analogies on imgs and captions

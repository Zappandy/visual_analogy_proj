import torch
import cv2
import os
from torch.utils.data import Dataset, DataLoader
from data_loaders import ImageDataset, visual_collate_fn
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from data_config import *

jpg_pttrn = ".jpg"
# images_1 https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
# https://medium.com/analytics-vidhya/how-to-load-any-image-dataset-in-python-3bd2fa2cb43d
# https://stackoverflow.com/questions/65279115/how-to-use-collate-fn-with-dataloaders
# https://medium.com/analytics-vidhya/how-to-load-any-image-dataset-in-python-3bd2fa2cb43d
visual_genome = ImageDataset(VG_PATH, pattern=jpg_pttrn)
flickr30k = ImageDataset(FLICKR_PATH, pattern=jpg_pttrn)
#print(visual_genome[:5])  # this slicing does not work, need to be loaded to data_loader 
vg_dataloader = DataLoader(visual_genome, batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=visual_collate_fn)
#print(visual_genome[0])
iter_loader = iter(vg_dataloader)
batch1 = next(iter_loader)
print(batch1)
batch2 = next(iter_loader)
print(batch2)
flickr_dataloader = DataLoader(flickr30k, batch_size=BATCH_SIZE, shuffle=False)
#print(visual_genome[:5])

#4 cap 4 images --> analogies on imgs and captions

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
raise SystemExit

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_step(image_paths):
    # image_paths --> should be dataloader now
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

x = predict_step(vg_dataloader[0])
